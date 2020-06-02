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
from common.utils.training_util import *
from AlexNet.transformation import *
import time
from tqdm import tqdm
import os
import logging
import logging.handlers
import sys
from apex import amp
import apex
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    def __init__(self, logger):
        self.logger = logger

    def info(self, message):
        self.logger.info(message)
        print(message)

    def error(self, message):
        self.logger.error(message)
        print(message)

    def warning(self, message):
        self.logger.warning(message)
        print(message)


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
        self.last_checkpoint_file = None
        self.load_from_check_point = None
        self.logger = None

        self.CHECKPOINT_PATH = None
        self.CHECKPOINT_INTERVAL = None
        self.NUM_CLASSES = None
        self.INPUT_DIR = None
        self.TRAIN_DIR = None
        self.VALID_DIR = None
        self.TRAIN_CSV = None
        self.VALID_CSV = None
        self.DEVICE = None
        self.EPOCHS = None
        self.PROJECT_NAME = None
        self.LOGFILE = None
        self.LOGLEVEL = None
        self.MULTI_GPU = None
        self.FP16_MIXED = None

    def init_logging(self):
        """
            Initialize the logger so that both console and file logging can be enabled
        """

        # Initialize logging
        log = logging.getLogger()
        log.setLevel(self.LOGLEVEL)
        formatter = logging.Formatter(logging.BASIC_FORMAT)

        # Create file handler
        file_handler = logging.handlers.WatchedFileHandler(self.LOGFILE)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

        # Initialize the logger instance
        self.logger = BaseLogger(log)

    def init_checkpoint(self):
        """
            The following logic is to automatically determine whether to load from
            the last checkpoint and determine the last checkpoint file.
        """

        # If the last.checkpoint file exists in the input dir
        if os.path.isfile(f'{self.INPUT_DIR}/last.checkpoint'):
            # Open the file and read the first line
            with open(f'{self.INPUT_DIR}/last.checkpoint', 'r') as file:
                self.last_checkpoint_file = file.readlines()[0]
                # If the file exists then load from last checkpoint
                if os.path.isfile(self.last_checkpoint_file):
                    self.load_from_check_point = True


class BaseExecutor(InitObject):

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

        # Initialize the tensor board summary writer
        self.tb_writer = SummaryWriter()

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
        if self.FP16_MIXED:
            # If mixed precision is enabled use the scaled loss for back prop
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
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
        os.makedirs(self.CHECKPOINT_PATH)

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

    def save_checkpoint(self, epoch):
        """
            This function is for saving model to disk.
        """

        # Create the checkpoint dir
        self.create_checkpoint_folder()

        file_name = f'{self.CHECKPOINT_PATH}/{self.PROJECT_NAME}_checkpoint_{epoch}.pth'
        self.logger.info(f"\tSaving checkpoint [{file_name}]...")

        # Create the checkpoint file
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # Add scheduler if available
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        # Save the checkpoint file to disk
        torch.save(checkpoint, file_name)

        # Indicate the last checkpoint file to last.checkpoint
        # This will be used for next run to load from checkpoint automatically
        with open(f'{self.INPUT_DIR}/last.checkpoint', 'w+') as file:
            file.write(file_name)

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

            # Push the parameters to the GPU
            self.logger.info(f"\tSuccessfully loaded model from checkpoint {self.last_checkpoint_file} ...")

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
