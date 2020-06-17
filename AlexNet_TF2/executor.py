import tensorflow as tf
from AlexNet_TF2.model import *
from common.tf.utils.base_executor import *

"""
    The Executor class is responsible for the training and testing of the AlexNet paper. It takes the data loaders and
    configuration (properties.py) as the input. This extends the parents class BaseExecutor, which 
    contains many boilerplate reusable methods.  

    This class was written to reduce and simply the lines of reusable codes needed for a functioning CNN.

"""


class Executor(BaseExecutor):
    def __init__(self, version, data_loaders, config):
        super().__init__(data_loaders, config)
        self.version = version

    def build_model(self):
        """
            This function is for instantiating the model, optimizer, learning rate scheduler and loss function.
        """
        # Enable Precision Mode before getting the model
        self.enable_precision_mode()

        # initialize the model
        self.model = AlexNetModel(num_classes=self.NUM_CLASSES)

        # self.enable_multi_gpu_training()

        # Initialize the Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=0.0001)

        # Define the Loss Function
        self.criterion = tf.keras.losses.sparse_categorical_crossentropy

        # Compile the Model
        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

    """
    pending:    
    4. Load/Save Custom Models
    5. Custom Training Loop
    
    """

    def train(self):
        """
            This function is used for training the network.
        """

        # Build the model
        self.call_build_model()

        # Define the checkpoint folder if needed
        self.define_checkpoint_folder()

        # Training Loop
        self.logger.info("Training starting now ...")

        # Any Keras supported values can be passed to the keras_fit_function() method.
        # 'validation_split' could be used in case separate validation data is not available, however cant be used with tf Dataset
        # 'class_weight' can also be used for imbalance datasets. Ex: { 0: 1., 1: 50., 2: 2.}. 1 has more weights and 0 and 2.
        # 'sample_weight' is for per training instance and should have same dimension as the training chunk.
        #       - If both sample_weights and class_weights are provided, the weights are multiplied together.
        self.keras_fit_function()

    def evaluate(self):
        # Build the model
        # self.logger.info("Building model ...")
        # self.build_model()

        # Load model from checkpoint
        self.load_checkpoint()

        # Involve the evaluate function
        self.model.evaluate(self.test_data_loader)

    def prediction(self):
        """
        This function is for predicting the rank 1 and rank 5 accuracy
        """

        # Build the model - Not required through
        # self.logger.info("Building model ...")
        # self.build_model()

        # Load model from checkpoint
        self.load_checkpoint(test=True)

        # Calculate the accuracy
        rank1_accuracy, rank5_accuracy = self.prediction_accuracy()

        self.logger.info(f"Test rank 1 Accuracy is {rank1_accuracy} and rank 5 accuracy is {rank5_accuracy}")

        return rank1_accuracy, rank5_accuracy
