import numpy as np
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from AlexNet.model import AlexNetModel
from AlexNet.properties import *
from common.dataset.dataset import ClassificationDataset
import pandas as pd
import matplotlib.pyplot as plt
from common.utils.logging_util import *
from common.utils.training_util import *
from AlexNet.transformation import *
import time
from tqdm import tqdm


class InitObject(object):
    def __init__(self):
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.loss_hist = None
        self.pbar = None

        self.CHECKPOINT_PATH = None
        self.NUM_CLASSES = None
        self.INPUT_DIR = None
        self.TRAIN_DIR = None
        self.VALID_DIR = None
        self.TRAIN_CSV = None
        self.VALID_CSV = None
        self.DEVICE = None
        self.EPOCHS = None


class BaseExecutor(InitObject):

    def __init__(self, data_loaders, config):
        super().__init__()
        self.__dict__.update(**config)
        self.train_data_loader = data_loaders['TRAIN'] if 'TRAIN' in data_loaders else None
        self.val_data_loader = data_loaders['VAL'] if 'VAL' in data_loaders else None
        self.test_data_loader = data_loaders['TEST'] if 'TEST' in data_loaders else None

        # Set loading from checkpoint to false
        self.load_from_check_point = False

        '''
        The following logic is to automatically determine whether to load from
        the last checkpoint and determine the last checkpoint file.
        '''
        # If the last.checkpoint file exists in the input dir
        if os.path.isfile(f'{self.INPUT_DIR}/last.checkpoint'):
            # Open the file and read the first line
            with open(f'{self.INPUT_DIR}/last.checkpoint', 'r') as file:
                self.last_checkpoint_file = file.readlines()[0]
                # If the file exists then load from last checkpoint
                if os.path.isfile(self.last_checkpoint_file):
                    self.load_from_check_point = True

    def forward_backward_pass(self, images, labels, epoch):
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
        self.loss_hist.send(loss.item())

        # compute gradients using back propagation
        loss.backward()

        # update parameters
        self.optimizer.step()

        # Update Progress Bar with the running average loss. Later add the validation accuracy
        self.pbar.set_postfix(epoch=f" {epoch}, loss= {round(self.loss_hist.value, 4)}", refresh=True)
        self.pbar.update()

    def calculate_validation_loss_accuracy(self):
        """
            This function is for calculating the validation loss and accuracy
        """

        # Set the model to eval mode.
        self.model.eval()

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
                outputs = self.model(images)

                # Output dimension is [ batch x num classes ].
                # Each value is a probability of the corresponding class
                # the torch.max() function is similar to the np.argmax() function,
                # where dim is the dimension to reduce.
                # outputs.data is not needed as from pytorch version 0.4.0
                # its no longer needed to access the underlying data of the tensor.
                # the first value in the tuple is the actual probability and
                # the 2nd value is the indexes corresponding to the max probability for that row
                # refer following url for more details
                # https://pytorch.org/docs/master/generated/torch.max.html
                # Example output : tuple([[0.42,0.56,0.86,0.45]],[[4,2,3,8]])
                _, predicted = torch.max(outputs, dim=1)

                # labels tensor is of dimension [ batch x 1 ],
                # hence labels.size(0) will provide the number of images / batch size
                total += labels.size(0)

                # Calculate corrected classified images.
                # both are 2D tensor hence can be compared against each other
                # Need to call .item() as both predicted and labels are 2D, the
                # resulting output of (predicted == labels).sum() is also 2D Tensor.
                # Hence to get a Python number from a tensor containing a single value,
                # we need to call .item()
                # Example: [[15]] => 15
                correct += (predicted == labels).sum().item()

        # calculate the accuracy in percentage
        accuracy = (100 * correct / total)
        return accuracy

    def create_checkpoint_folder(self):
        """
            This function is for creating an empty checkpoint folder. The folder does not get created until
            there is a checkpoint to save.
        """
        self.CHECKPOINT_PATH = f'{self.INPUT_DIR}/checkpoint/{datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")}'
        # os.mkdir(self.CHECKPOINT_PATH)

    def init_training_loop(self):
        """
            This function is for defining common steps before starting the each training loop.
        """

        # Set model to training mode
        self.model.train()

        # Reset the average losses
        self.loss_hist.reset()

        # Initialize the progress bar
        self.pbar = tqdm(total=len(self.train_data_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)


class Executor(BaseExecutor):
    def __init__(self, version, data_loaders, config):
        super().__init__(data_loaders, config)
        self.version = version

    def build_model(self):
        """
            This function is for instantiating the model, optimizer, learning rate scheduler and loss function.
        """

        # initialize the model
        self.model = AlexNetModel(num_classes=self.NUM_CLASSES)

        # Send the model to GPU
        self.model.to(self.DEVICE)

        # Initialize the Optimizer
        # Need to call self.model.model.parameters() as model is an attribute in the Module function.
        self.optimizer = torch.optim.SGD(self.model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)

        # Initialize the learning rate scheduler
        # Reduce learning rate when a metric has stopped improving.
        # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
        # This scheduler reads a metrics quantity and if no improvement is seen
        # for a ‘patience’ number of epochs, the learning rate is reduced.
        # https://pytorch.org/docs/stable/optim.html?highlight=reducelronplateau#torch.optim.lr_scheduler.ReduceLROnPlateau
        # Here val accuracy has been used as the metric, hence set mode to 'max'
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # Define the Loss Function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Push the parameters to the GPU
        self.model.cuda()

        # Loss Average
        self.loss_hist = AverageLoss()

    def train(self):
        """
            This function is used for training the network.
        """

        self.build_model()

        for epoch in range(self.EPOCHS):

            self.init_training_loop()

            for i, (images, labels, _) in enumerate(self.train_data_loader):
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                self.forward_backward_pass(images, labels, epoch)

            eval_accuracy = self.calculate_validation_loss_accuracy()
            self.scheduler.step(eval_accuracy)
            self.pbar.set_postfix(epoch=f" {epoch}, loss = {round(self.loss_hist.value, 4)}, val acc= {round(eval_accuracy, 3)}", refresh=False)
            self.pbar.close()


if __name__ == '__main__':
    def getDataLoader(csv_path, images_path, transformation, fields, training=False, batch_size=16, shuffle=False, num_workers=4, pin_memory=False,
                      drop_last=True):
        df = pd.read_csv(csv_path)
        dataset = ClassificationDataset(images_path, df, transformation, fields, training)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        return data_loader


    fields = {'image': 'image', 'label': 'class'}
    train_data_loader = getDataLoader(csv_path=config['TRAIN_CSV'], images_path=config['TRAIN_DIR'], transformation=train_transformation,
                                      fields=fields,
                                      training=True,
                                      batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
    val_data_loader = getDataLoader(csv_path=config['VALID_CSV'], images_path=config['VALID_DIR'], transformation=test_transformation, fields=fields,
                                    training=False,
                                    batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    e = Executor("", {'TRAIN': train_data_loader, 'VAL': val_data_loader}, config=config)
    e.train()
