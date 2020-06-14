import torch
from common.tf.utils.training_util import *
from tqdm import tqdm
from apex import amp
from common.tf.utils.init_executor import *
import numpy as np

"""
    This class was written to reduce and simply the lines of reusable codes needed for a functioning 
    CNN.
    
    This is the very initial version, hence the BaseExecutor class is expected to evolve with more 
    complex functionality.
        
    There are many aspects which are covered in the Executor and its parent class BaseExecutor, such as:
        1.  Data Augmentation is outside of this class and can be defined in a 
            semi declarative way using albumentations library inside the transformation.py class.
        2.  Automatic Loading and Saving models from and to checkpoint. 
        3.  Integration with Tensor Board. The Tensor Board data is being written after a checkpoint save.
            This is to make sure that upon restarting the training, the plots are properly drawn.
                A.  Both Training Loss and Validation Accuracy is being written. The code will be modified to 
                    also include Training Accuracy and Validation Loss.
                B.  The model is also being stored as graph for visualization.
        4.  Logging has been enabled in both console and external file. The external file name can be configured 
            using the configuration.
        5.  Multi-GPU Training has been enabled using torch.nn.DataParallel() functionality. 
        6.  Mixed Precision has been enabled using Nvidia's apex library as the PyTorch 1.6 is not released yet.
            None:   At this moment both Multi-GPU and Mixed Precision can not be using together. This will be fixed 
                    once PyTorch 1.6 has been released.
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
        self.learning_rate = []

    def forward_backward_pass(self, images, labels, epoch, i):
        """
            This function is for one time forward and backward pass. The epoch is used only for logging.

            :argument

            :param images
            :param labels
            :param epoch

        """
        # Empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # Forward Pass
        # output dimension is [ batch x num classes ].
        output = self.model(images)

        # Compute Loss
        # Need to call squeeze() on the labels tensor to
        # returns a tensor with all the dimensions of input of size 1 removed. ( 2D -> 1D )
        # labels has the dimension of [ batch x 1 ] ( 2D Tensor )
        # labels.squeeze() will have the dimension of [ batch ] ( 1D Tensor )
        loss = self.criterion(output, labels.squeeze())

        # Calculate the average loss
        self.train_loss_hist.send(loss.item())

        # compute gradients using back propagation
        if self.FP16_MIXED:
            # If mixed precision is enabled use the scaled loss for back prop
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # update parameters
        self.optimizer.step()

        # Update Progress Bar with the running average loss. Later add the validation accuracy
        self.pbar.set_postfix(epoch=f" {epoch}, loss= {round(self.train_loss_hist.value, 4)}", refresh=True)
        self.pbar.update()

        return output

    def calculate_validation_loss_accuracy(self):
        """
            This function is for calculating the validation loss and accuracy
        """

        # Set the model to eval mode.
        self.model.eval()

        self.val_loss_hist = AverageLoss()

        # global variable
        correct = 0
        total = 0

        # with no gradient mode on
        with torch.no_grad():
            # Loop through the validation data loader
            for images, labels, _ in self.val_data_loader:
                # Move the tensors to GPU
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                # Forward pass
                predictions = self.model(images)

                # Calculate the loss
                loss = self.criterion(predictions, labels.squeeze())

                total, correct = self.cal_prediction(predictions, labels, total, correct)

                # Calculate the average loss
                self.val_loss_hist.send(loss.item())

        # calculate the accuracy in percentage
        accuracy = (100 * correct / total)
        return accuracy

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

    def create_checkpoint_folder(self):
        """
            This function is for creating an empty checkpoint folder. The folder does not get created until
            there is a checkpoint to save.
        """
        self.CHECKPOINT_PATH = f'{self.INPUT_DIR}/checkpoint/{datetime.now().strftime("%b-%d-%Y-%H-%M-%S")}'

    def pre_training_loop_ops(self):
        """
            This function is for defining common steps before starting the each training loop.
        """

        # Set model to training mode
        self.model.train()

        # Reset the average losses
        self.train_loss_hist.reset()

        # Initialize the progress bar
        self.pbar = tqdm(total=len(self.train_data_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

    def post_training_loop_ops(self, epoch, train_accuracy):
        # add to train loss
        self.train_loss.append((round(self.train_loss_hist.value, 4), epoch))

        # Train Accuracy for the epoch
        self.train_acc.append((round(train_accuracy, 3), epoch))

        eval_accuracy = self.calculate_validation_loss_accuracy()

        current_lr = self.get_lr()

        # Display the validation loss/accuracy in the progress bar
        self.pbar.set_postfix(
            epoch=f"{epoch}, loss={round(self.train_loss_hist.value, 4)}, val acc={round(eval_accuracy, 3)}, train acc={round(train_accuracy, 3)}, lr={current_lr}",
            refresh=False)

        self.logger.info(
            f"epoch={epoch}, loss={round(self.train_loss_hist.value, 4)}, val acc={round(eval_accuracy, 3)}, train acc={round(train_accuracy, 3)}, lr={current_lr}")

        # Add to validation loss
        self.val_acc.append((round(eval_accuracy, 3), epoch))
        self.val_loss.append((round(self.val_loss_hist.value, 4), epoch))

        self.learning_rate.append((current_lr, epoch))

        # Save the model ( if needed )
        self.save_checkpoint(epoch)

        return eval_accuracy

    def save_checkpoint(self, epoch):
        """
            This function is for saving model to disk.
        """

        if epoch % self.CHECKPOINT_INTERVAL == 0:
            # Create the checkpoint dir
            if not os.path.isdir(self.CHECKPOINT_PATH):
                os.makedirs(self.CHECKPOINT_PATH)

            file_name = f'{self.CHECKPOINT_PATH}/{self.PROJECT_NAME}_checkpoint_{epoch}.pth'
            self.logger.info(f"\n\tSaving checkpoint [{file_name}]...")

            # Create the checkpoint file
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }

            # Add scheduler if available
            if self.scheduler:
                checkpoint['scheduler'] = self.scheduler.state_dict()

            # Save the checkpoint file to disk
            torch.save(checkpoint, file_name)

            # Indicate the last checkpoint file to last.checkpoint
            # This will be used for next run to load from checkpoint automatically
            with open(f'{self.INPUT_DIR}/last.checkpoint.{self.PROJECT_NAME}', 'w+') as file:
                file.writelines('\n'.join([file_name, self.tb_writer.get_logdir()]))

            # Write to tensor board
            for (tr_y, tr_x), (val_y, val_x) in zip(self.train_loss, self.val_loss):
                self.tb_writer.add_scalars("Loss", {'train': tr_y, 'val': val_y}, tr_x)

            for (tr_y, tr_x), (val_y, val_x) in zip(self.train_acc, self.val_acc):
                self.tb_writer.add_scalars("Accuracy", {'train': tr_y, 'val': val_y}, tr_x)

            for y, x in self.learning_rate:
                self.tb_writer.add_scalar("Learning Rate", y, x)

            # Reset the arrays
            self.init_temp_arrays_for_tensor_board()

    def load_checkpoint(self):
        """
            This function is for loading model from checkpoint.
        """
        start_epoch = 1

        if self.load_from_check_point:
            self.logger.info(f"\tAttempting to load from checkpoint {self.last_checkpoint_file} ...")

            checkpoint = torch.load(self.last_checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

            start_epoch = checkpoint['epoch'] + 1

            self.logger.info(f"\tSuccessfully loaded model from checkpoint {self.last_checkpoint_file} ...")

        # Push the parameters to the GPU
        self.model.cuda()

        return start_epoch

    def enable_multi_gpu_training(self):
        """
            This function is for using multiple GPUs in one system for training
        """
        if self.MULTI_GPU:
            # Verify if there are more then 1 GPU
            if torch.cuda.device_count() > 1:
                self.logger.info(f"\tUsing {int(torch.cuda.device_count())} GPUs for training ...")

                # Use torch.nn.DataParallel()
                if self.FP16_MIXED:
                    # self.model = apex.parallel.DistributedDataParallel(self.model)
                    self.logger.error("FP16_MIXED can not be used with MULTI_GPU enabled")
                    raise Exception("Compatibility error, please review logs ...")
                else:
                    self.model = torch.nn.DataParallel(self.model)

    def enable_precision_mode(self):
        """
            This function is for using FP16 with Mixed Precision.
            As of now PyTouch 1.6 is not stable which as enabled
            Mixed Precision training without Nvidia library.
            Hence using amp library, the downside is it cannot be used
            with multiple GPU just yet.

            https://nvidia.github.io/apex/amp.html
        """
        if self.FP16_MIXED:
            # '00' : FP32 training
            # '01' : Mixed Precision (recommended for typical use)
            # '02' : “Almost FP16” Mixed Precision
            # '03' : FP16 training
            opt_level = 'O1'
            # Use amp for mixed precision mode
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

            self.logger.info(f"\tMixed Precision mode enabled for training ...")

    def save_model_to_tensor_board(self):
        """
            This function is for saving the graph to tensor board
        """

        # load a batch from the validation data loader
        loader = iter(self.val_data_loader)
        images, labels, _ = loader.next()
        if self.tb_writer is None:
            now = datetime.now()
            self.tb_writer = SummaryWriter(log_dir=f'runs/{self.PROJECT_NAME}_{now.strftime("%Y%m%d-%H%M%S")}')

        # save the model graph to tensor board
        self.tb_writer.add_graph(self.model, images)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def cal_prediction(predictions, labels, total, correct):
        # Output dimension is [ batch x num classes ].
        # Each value is a probability of the corresponding class
        # the torch.max() function is similar to the np.argmax() function,
        # where dim is the dimension to reduce.
        # outputs.data is not needed as from torch version 0.4.0
        # its no longer needed to access the underlying data of the tensor.
        # the first value in the tuple is the actual probability and
        # the 2nd value is the indexes corresponding to the max probability for that row
        # refer following url for more details
        # https://pytorch.org/docs/master/generated/torch.max.html
        # Example output : tuple([[0.42,0.56,0.86,0.45]],[[4,2,3,8]])
        _, predicted = torch.max(predictions, dim=1)

        # labels tensor is of dimension [ batch x 1 ],
        # hence labels.size(0) will provide the number of images / batch size
        total += labels.size(0)

        # Reduce the dimension of labels from [batch x 1] -> [batch]
        labels = labels.squeeze()

        correct += (predicted == labels).sum().item()

        return total, correct

    @staticmethod
    def rank5_accuracy(predictions, labels, total, correct):

        _, predicted = torch.topk(predictions, k=5, dim=1)

        # labels tensor is of dimension [ batch x 1 ],
        # hence labels.size(0) will provide the number of images / batch size
        total += labels.size(0)

        # Reduce the dimension of labels from [batch x 1] -> [batch]
        labels = labels.squeeze()

        predicted = predicted.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for i in range(labels.shape[0]):
            if labels[i] in predicted[i]:
                correct += 1

        return total, correct

    def keras_fit_function(self, params={}):

        fit_params = {
            'epochs': self.EPOCHS,
            'steps_per_epoch': int(self.TRAIN_DATA_SIZE / self.TRAIN_BATCH_SIZE),
            'validation_steps': int(self.VAL_DATA_SIZE / self.VAL_BATCH_SIZE)
        }

        fit_params.update(params)

        history = self.model.fit(self.train_data_loader, validation_data=self.val_data_loader, **fit_params)
