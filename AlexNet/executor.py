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
import timeit
from common.utils.base_executor import *


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

        self.enable_multi_gpu_training()

        # Send the model to GPU
        self.model.to(self.DEVICE)

        # Initialize the Optimizer
        # Need to call self.model.model.parameters() as model is an attribute in the Module function.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)

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

        # Loss Average
        self.loss_hist = AverageLoss()

        # Enable Precision Mode
        self.enable_precision_mode()

    def train(self):
        """
            This function is used for training the network.
        """

        # Build the model
        self.logger.info("Building model ...")
        self.build_model()

        # Load model from checkpoint if needed
        start_epoch = self.load_checkpoint()

        # Training Loop
        self.logger.info("Training starting now ...")
        for epoch in range(start_epoch, self.EPOCHS):

            self.init_training_loop()

            for i, (images, labels, _) in enumerate(self.train_data_loader):
                images = images.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                self.forward_backward_pass(images, labels, epoch)

                self.tb_writer.add_scalar("Loss/train", round(self.loss_hist.value, 4), (epoch - 1) * len(self.train_data_loader) + i)

            eval_accuracy = self.calculate_validation_loss_accuracy()

            # Scheduler step() function
            self.scheduler.step(eval_accuracy)

            # Display the validation loss/accuracy in the progress bar
            self.pbar.set_postfix(epoch=f" {epoch}, loss = {round(self.loss_hist.value, 4)}, val acc= {round(eval_accuracy, 3)}", refresh=False)

            # Close the progress bar
            self.pbar.close()

            # Save the model ( if needed )
            self.save_checkpoint(epoch)

            self.tb_writer.add_scalar("Accuracy/val", round(eval_accuracy, 3), epoch)

        self.tb_writer.close()


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

    start = timeit.default_timer()
    e.train()
    stop = timeit.default_timer()
    print('Time: ', stop - start)