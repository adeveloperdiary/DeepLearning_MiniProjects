import numpy as np
import cv2

import torch
import torch.nn as nn
from common.utils.model_util import *

"""
    This is the implementation of the VGG Architecture using PyTorch Library  
    There are few differences between the Actual Paper and this implementation.
        
    1.  Use of Batch Normalization after the activation layer     
    3.  Use more Dropout layers ( after MaxPool layers ) to reduce over-fitting.
    4.  Use Xavier Normal initialization instead of initializing just from a normal distribution and use that to initialize in retraining 
        with more depth in the architecture ( A -> B -> D -> E ). 
    
    Some of the below code was taken from pytorch code base.
             
"""


class VGG(CNNBaseModel):
    # Define different configurations of VGG Network
    vgg_configs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, network_type='A', num_classes=256):
        """
            This constructor is responsible for defining the layers in the architecture.
        """
        super(VGG, self).__init__()

        # select the network type
        selected_network = self.vgg_configs[network_type]

        layers = []

        # Assuming color images are used, hence the incoming channel is 3
        in_channels = 3

        # Loop through the array and create layers as needed
        for out_channels in selected_network:
            # Create MaxPool2d for M
            if out_channels == 'M':

                # Add MaxPool2d and Dropout layer. Dropout layer was not used by the authors.
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [torch.nn.Dropout(p=0.25)]
            else:

                # Add the convolution, relu and batch norm layer
                layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)]
                layers += [torch.nn.PReLU()]
                layers += [torch.nn.BatchNorm2d(out_channels)]
                in_channels = out_channels

        # flatten the layer
        layers += [Flatten()]

        # Add the fully connected layers
        layers += [torch.nn.Linear(7 * 7 * 512, 4096)]
        layers += [torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Dropout(p=0.5)]
        layers += [torch.nn.Linear(4096, 4096)]
        layers += [torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Dropout(p=0.5)]
        layers += [torch.nn.Linear(4096, num_classes)]

        # dim - A dimension along which LogSoftmax will be computed.
        # Since our inout is (N, L), we need to pass 1
        layers += [torch.nn.LogSoftmax(dim=1)]

        self.model = torch.nn.Sequential(*layers)

        # Use xavier normal initializer
        self.weights_init_xavier_normal()

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.model(x)
        return x

    def print_network(self):
        """
            Use this function to print the network sizes.
        """
        X = torch.rand(1, 3, 224, 224)
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__, 'Output shape:\t', X.shape)
