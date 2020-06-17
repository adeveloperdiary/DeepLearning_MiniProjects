import torch
from common.tf.utils.training_util import *
from tqdm import tqdm
from apex import amp
from common.tf.utils.init_executor import *
import numpy as np
import tensorflow as tf
from common.tf.callbacks.checkpoint import *
from datetime import datetime

"""
    This class was written to reduce and simply the lines of reusable codes needed for a functioning 
    CNN.
    
    This is the very initial version, hence the BaseExecutor class is expected to evolve with more 
    complex functionality.
        
    There are many aspects which are covered in the Executor and its parent class BaseExecutor, such as:
        1.  Data Augmentation is outside of this class transformation.py class.
        2.  Automatic Loading and Saving models from and to checkpoint. 
        3.  Integration with Tensor Board. The Tensor Board data is being written after a checkpoint save.
            This is to make sure that upon restarting the training, the plots are properly drawn.
                A.  Both Training Loss and Validation Accuracy is being written. The code will be modified to 
                    also include Training Accuracy and Validation Loss.
                B.  The model is also being stored as graph for visualization.
        4.  Logging has been enabled in both console and external file. The external file name can be configured 
            using the configuration.
        5.  Multi-GPU Training has been enabled using torch.nn.DataParallel() functionality. 
        6.  Mixed Precision has been enabled using tf.keras.mixed_precision.experimental.set_policy() function.
"""


class BaseExecutor(InitExecutor):

    def __init__(self, data_loaders, config):
        super().__init__()
        self.__dict__.update(**config)
        self.train_data_loader = data_loaders['TRAIN'] if 'TRAIN' in data_loaders else None
        self.val_data_loader = data_loaders['VAL'] if 'VAL' in data_loaders else None
        self.test_data_loader = data_loaders['TEST'] if 'TEST' in data_loaders else None

        # Set loading from checkpoint to false
        self.load_from_check_point = False

        # Initialize Logging
        self.init_logging()

        # Init Checkpoint
        self.init_checkpoint()

        # Initialize the arrays for tensor boards
        # This is needed as scalar data will be pushed to tensor board
        # only when the checkpoint is being saved.
        # So that in the next run the plot will be properly appended.
        self.init_temp_arrays_for_tensor_board()

    def init_temp_arrays_for_tensor_board(self):
        # Reset the temp lists
        self.val_acc = []
        self.val_loss = []
        self.train_loss = []
        self.train_acc = []
        self.learning_rate_cache = []

    def cache_loss_accuracy_per_epoch(self, epoch, logs):

        self.train_loss.append((round(logs.get('loss'), 4), epoch))

        # Train Accuracy for the epoch
        self.train_acc.append((round(logs.get('accuracy'), 3), epoch))

        self.val_loss.append((round(logs.get('val_loss'), 4), epoch))

        # Train Accuracy for the epoch
        self.val_acc.append((round(logs.get('val_accuracy'), 3), epoch))

        self.learning_rate_cache.append((self.model.optimizer.learning_rate.numpy(), epoch))

    def prediction_accuracy(self):
        """
            This function is for predicting test accuracy
        """

        # Create int variable for storing the correct prediction
        correct_rank1 = 0
        correct_rank5 = 0

        # Invoke the predict_proba() method, since Dataset is being used, batch_size needs to be set as None.
        # The predict_proba() method returns a NumPy array ( not Tensor ) of size [ # of Samples x # of Classes ]
        predicted = self.model.predict_proba(self.test_data_loader, batch_size=None)

        # Find the total number of test sample
        total = predicted.shape[0]

        # Create a list for string the target class labels.
        target = []

        # Iterate the dataset object and store the target class labels.
        for image, label in self.test_data_loader:
            target.extend(label.numpy().squeeze().tolist())

        # Combine the predicted and target class labels to calculate the accuracies
        for t, p in zip(target, predicted):

            # First use argsort to get the class labels then reverse it to arrange from height to lowest
            p = np.argsort(p)[::-1]

            # Rank 5 Accuracy
            if t in p[:5]:
                correct_rank5 += 1

            # Rank 1 Accuracy
            if t == p[0]:
                correct_rank1 += 1

        # Return the values
        return correct_rank1 / total, correct_rank5 / total

    def define_checkpoint_folder(self):
        """
            This function is for creating an empty checkpoint folder. The folder does not get created until
            there is a checkpoint to save.
        """
        self.CHECKPOINT_PATH = f'{self.INPUT_DIR}/checkpoint/{datetime.now().strftime("%b-%d-%Y-%H-%M-%S")}'

    def save_checkpoint(self, epoch, logs):
        """
            This function is for saving model to disk.
        """

        self.cache_loss_accuracy_per_epoch(epoch, logs)

        if (epoch + 1) % self.CHECKPOINT_INTERVAL == 0:
            # Create the checkpoint dir
            if not os.path.isdir(self.CHECKPOINT_PATH):
                os.makedirs(self.CHECKPOINT_PATH)

            file_name = f'{self.CHECKPOINT_PATH}/{self.PROJECT_NAME}_checkpoint_{epoch}.h5'
            self.logger.info(f"\n\tSaving checkpoint [{file_name}]...")

            self.model.save(file_name)

            # Indicate the last checkpoint file to last.checkpoint
            # This will be used for next run to load from checkpoint automatically
            with open(f'{self.INPUT_DIR}/last.checkpoint.{self.PROJECT_NAME}', 'w+') as file:
                file.writelines('\n'.join([file_name, self.tb_writer.get_logdir()]))

            # Write to tensor board
            for (tr_y, tr_x), (val_y, val_x) in zip(self.train_loss, self.val_loss):
                self.tb_writer.add_scalars("Loss", {'train': tr_y, 'val': val_y}, tr_x)

            for (tr_y, tr_x), (val_y, val_x) in zip(self.train_acc, self.val_acc):
                self.tb_writer.add_scalars("Accuracy", {'train': tr_y, 'val': val_y}, tr_x)

            for y, x in self.learning_rate_cache:
                self.tb_writer.add_scalar("Learning Rate", y, x)

            # Reset the arrays
            self.init_temp_arrays_for_tensor_board()

    def load_checkpoint(self, test=False):
        """
            This function is for loading model from checkpoint.
        """
        start_epoch = 0

        if self.load_from_check_point:
            self.logger.info(f"\tAttempting to load from checkpoint {self.last_checkpoint_file} ...")

            self.model = tf.keras.models.load_model(self.last_checkpoint_file)

            # checkpoint = torch.load(self.last_checkpoint_file)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # if 'scheduler' in checkpoint:
            #    self.scheduler.load_state_dict(checkpoint['scheduler'])

            # start_epoch = checkpoint['epoch'] + 1
            start_epoch = int(self.last_checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])

            if self.EPOCHS <= start_epoch + 1 and not test:
                raise Exception('[ERROR] Epoch is smaller than current epoch ...', 'Please change the epoch in properties.py')

            self.logger.info(f"\tSuccessfully loaded model from checkpoint {self.last_checkpoint_file} ...")

        return start_epoch

    def enable_multi_gpu_training(self):
        """
            This function is for using multiple GPUs in one system for training
        """
        if self.MULTI_GPU:
            # Verify if there are more then 1 GPU
            no_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
            print("Num GPUs Available: ", no_gpu)
            if no_gpu > 1:
                tf.debugging.set_log_device_placement(True)
                self.multi_gpu_strategy = tf.distribute.MirroredStrategy()
                return True
        return False

    def enable_precision_mode(self):
        """
            This function is for using FP16 with Mixed Precision.
        """
        if self.FP16_MIXED:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)
            self.logger.info(
                f"\tMixed Precision mode enabled for training... [ Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype} ]")

    def keras_fit_function(self, params={}):

        # Load model from checkpoint if needed
        start_epoch = self.load_checkpoint()

        fit_params = {
            'epochs': self.EPOCHS,
            'steps_per_epoch': int(self.TRAIN_DATA_SIZE / self.TRAIN_BATCH_SIZE),
            'validation_steps': int(self.VAL_DATA_SIZE / self.VAL_BATCH_SIZE),
            'initial_epoch': start_epoch
        }

        fit_params.update(params)

        self.checkpoint = CustomCheckpoint(save_fn=self.save_checkpoint)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'runs/{self.PROJECT_NAME}', write_grads=True,
                                                              write_images=True)

        if self.LEARNING_RATE > 0:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.LEARNING_RATE)

        history = self.model.fit(self.train_data_loader, validation_data=self.val_data_loader, callbacks=[self.checkpoint, tensorboard_callback],
                                 **fit_params)

    def call_build_model(self, model_fn):
        """
            This function is for wrapping the model in multi_gpu_strategy.
        """
        if self.MULTI_GPU and self.enable_multi_gpu_training():
            with self.multi_gpu_strategy.scope():
                model_fn()
        else:
            model_fn()
