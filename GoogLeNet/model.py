import numpy as np
import cv2

import torch
import torch.nn as nn
from common.utils.model_util import *

"""
    This is the implementation of the GoogLeNet Architecture using PyTorch Library  
    There are few differences between the Actual Paper and this implementation.
                     
    1.  Use Xavier Normal initialization instead of initializing just from a normal distribution.
    2.  The auxiliary outputs are not implemented.  
    
    Some of the below code was taken from pytorch code base.
             
"""


class DefaultConvolutionModule(CNNBaseModel):
    """
        This class is for defining the following 3 layers together. It extends the CNNBaseModel parent class.
            1. Conv2d
            2. BatchNorm2d
            3. ReLU

        The only purpose of this class is to reduce number of coding lines.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        """
            The constructor of the DefaultConvolutionModule class.
            :param in_channels: number of input channel
            :param out_channels: output channel
            :param kernel: size of the kernel
            :param stride: size of stride
            :param padding: size of padding
        """
        super(DefaultConvolutionModule, self).__init__()

        # Define the convolution layer. The output spacial dimension remains same after convolution.
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)

        # Define the BatchNorm2d
        # eps – a value added to the denominator for numerical stability
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels, eps=0.001)

        # Define ReLU Activation.
        self.relu = torch.nn.ReLU(inplace=True)

        # Initialize the layers using xavier normal
        self.weights_init_xavier_normal()

    def forward(self, x):
        """
            The forward function for the DefaultConvolutionModule

            :param x: input data
        """
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionModule(CNNBaseModel):
    """
        This class defines the Inception Module for the GoogLeNet Architecture.
    """

    def __init__(self, in_channels, num1x1, num3x3reduce, num3x3, num5x5reduce, num5x5, num1x1reduce):
        """
            The constructor of the InceptionModule class.

            :param in_channels: number of input channel
            :param num1x1: channel size of 1x1 convolution
            :param num3x3reduce: bottleneck channel size of 3x3 convolution
            :param num3x3: channel size of 3x3 convolution
            :param num5x5reduce: bottleneck channel size of 5x5 convolution
            :param num5x5: channel size of 5x5 convolution
            :param num1x1reduce: bottleneck channel size of 1x1 convolution after max pool
        """
        super(InceptionModule, self).__init__()

        self.conv_1x1 = torch.nn.Sequential(
            DefaultConvolutionModule(in_channels=in_channels, out_channels=num1x1, kernel=1)
        )

        self.conv_3x3 = torch.nn.Sequential(
            DefaultConvolutionModule(in_channels=in_channels, out_channels=num3x3reduce, kernel=1),
            DefaultConvolutionModule(in_channels=num3x3reduce, out_channels=num3x3, kernel=3, padding=1)
        )

        self.conv_5x5 = torch.nn.Sequential(
            DefaultConvolutionModule(in_channels=in_channels, out_channels=num5x5reduce, kernel=1),
            DefaultConvolutionModule(in_channels=num5x5reduce, out_channels=num5x5, kernel=3, padding=1)
        )

        self.pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            DefaultConvolutionModule(in_channels=in_channels, out_channels=num1x1reduce, kernel=3, padding=1)
        )

    def forward(self, x):
        """
            The forward function for the DefaultConvolutionModule

            :param x: input data
        """
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)
        conv_5x5 = self.conv_5x5(x)
        pool = self.pool(x)

        # Combine the layers in array
        output = [conv_1x1, conv_3x3, conv_5x5, pool]

        # Concatenate by channel, hence dim=1 (N, C, H, W)
        return torch.cat(tensors=output, dim=1)


class GoogLeNet(CNNBaseModel):
    """
        This class defines the GoogLeNet Architecture.
    """

    def __init__(self, num_classes=256):
        """
            This constructor is responsible for defining the layers in the architecture.
        """
        super(GoogLeNet, self).__init__()
        self.model = torch.nn.Sequential(

            # Initial layers are same as AlexNet
            DefaultConvolutionModule(in_channels=3, out_channels=64, kernel=7, padding=3, stride=2),
            # Either have padding=1 or ceil_mode=True
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            DefaultConvolutionModule(in_channels=64, out_channels=64, kernel=1),
            DefaultConvolutionModule(in_channels=64, out_channels=192, kernel=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # Inception 3a
            InceptionModule(in_channels=192, num1x1=64, num3x3reduce=96, num3x3=128, num5x5reduce=16, num5x5=32, num1x1reduce=32),
            # Inception 3b
            InceptionModule(in_channels=256, num1x1=128, num3x3reduce=128, num3x3=192, num5x5reduce=32, num5x5=96, num1x1reduce=64),

            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # Inception 4a
            InceptionModule(in_channels=480, num1x1=192, num3x3reduce=96, num3x3=208, num5x5reduce=16, num5x5=48, num1x1reduce=64),
            # Inception 4b
            InceptionModule(in_channels=512, num1x1=160, num3x3reduce=112, num3x3=224, num5x5reduce=24, num5x5=64, num1x1reduce=64),
            # Inception 4c
            InceptionModule(in_channels=512, num1x1=128, num3x3reduce=128, num3x3=256, num5x5reduce=24, num5x5=64, num1x1reduce=64),
            # Inception 4d
            InceptionModule(in_channels=512, num1x1=112, num3x3reduce=144, num3x3=288, num5x5reduce=32, num5x5=64, num1x1reduce=64),
            # Inception 4e
            InceptionModule(in_channels=528, num1x1=256, num3x3reduce=160, num3x3=320, num5x5reduce=32, num5x5=128, num1x1reduce=128),

            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),

            # Inception 5a
            InceptionModule(in_channels=832, num1x1=256, num3x3reduce=160, num3x3=320, num5x5reduce=32, num5x5=128, num1x1reduce=128),
            # Inception 5b
            InceptionModule(in_channels=832, num1x1=384, num3x3reduce=192, num3x3=384, num5x5reduce=48, num5x5=128, num1x1reduce=128),

            # AdaptiveAvgPool2d will shrink the layers automatically. No need to provide the kernel.
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Dropout(p=0.4),

            # Flatten the layers
            Flatten(),

            # Define linear layers
            nn.Linear(1024, num_classes),
            # dim - A dimension along which LogSoftmax will be computed.
            # Since our inout is (N, L), we need to pass 1
            torch.nn.LogSoftmax(dim=1)
        )

        # Use xavier normal initializer
        self.weights_init_xavier_normal()

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        """
            The forward function for the GoogLeNet

            :param x: input data
        """
        x = self.model(x)
        return x
