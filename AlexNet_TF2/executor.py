import tensorflow as tf
from AlexNet_TF2.model import *
from common.tf.utils.base_executor import *

"""
    The Executor class is responsible for the training and testing of the AlexNet paper. It takes the data loaders and
    configuration (properties.py) as the input. This extends the parents class BaseExecutor, which 
    contains many boilerplate reusable methods.  

    This class was written to reduce and simply the lines of reusable codes needed for a functioning 
    CNN.

"""


class Executor(BaseExecutor):
    def __init__(self, version, data_loaders, config):
        super().__init__(data_loaders, config)
        self.version = version

    def build_model(self):
        """
            This function is for instantiating the model, optimizer, learning rate scheduler and loss function.
        """

        # Enable Precision Mode
        self.enable_precision_mode()

        # initialize the model
        self.model = AlexNetModel(num_classes=self.NUM_CLASSES)
        # Save model to tensor board
        # self.save_model_to_tensor_board()

        # self.enable_multi_gpu_training()

        # Initialize the Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=0.0005)

        # Initialize the learning rate scheduler
        # Reduce learning rate when a metric has stopped improving.
        # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
        # This scheduler reads a metrics quantity and if no improvement is seen
        # for a ‘patience’ number of epochs, the learning rate is reduced.
        # https://pytorch.org/docs/stable/optim.html?highlight=reducelronplateau#torch.optim.lr_scheduler.ReduceLROnPlateau
        # Here val accuracy has been used as the metric, hence set mode to 'max'
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=False)

        # Define the Loss Function
        self.criterion = tf.keras.losses.sparse_categorical_crossentropy

        # Loss Average
        # self.train_loss_hist = AverageLoss()

        # Compile the Model
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

    """
    pending:
    1. Add Callbacks:
        - lr scheduler
    2. Multi GPU Training
    4. Load/Save Custom Models
    5. Custom Training Loop
    
    """

    def train(self):
        """
            This function is used for training the network.
        """

        # Build the model
        self.logger.info("Building model ...")
        self.build_model()

        self.create_checkpoint_folder()

        # Training Loop
        self.logger.info("Training starting now ...")

        # Any Keras supported values can be passed to the keras_fit_function() method.
        # 'validation_split' could be used in case separate validation data is not available, however cant be used with tf Dataset
        # 'class_weight' can also be used for imbalance datasets. Ex: { 0: 1., 1: 50., 2: 2.}. 1 has more weights and 0 and 2.
        # 'sample_weight' is for per training instance and should have same dimension as the training chunk.
        #       - If both sample_weights and class_weights are provided, the weights are multiplied together.
        self.keras_fit_function()

    def evaluate(self):
        self.logger.info("Building model ...")
        self.build_model()

        # Load model from checkpoint
        self.load_checkpoint()

        self.model.evaluate(self.test_data_loader)

    def prediction(self):
        '''
        This function is for predicting the rank 1 and rank 5 accuracy
        '''

        self.logger.info("Building model ...")
        self.build_model()

        # Load model from checkpoint
        self.load_checkpoint(test=True)

        # Calculate the accuracy
        rank1_accuracy, rank5_accuracy = self.prediction_accuracy()

        self.logger.info(f"Test rank 1 Accuracy is {rank1_accuracy} and rank 5 accuracy is {rank5_accuracy}")

        return rank1_accuracy, rank5_accuracy
