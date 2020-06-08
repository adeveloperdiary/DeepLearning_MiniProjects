import numpy as np
import cv2

import torch
import torch.nn as nn
from common.utils.model_util import *

"""
    This is the implementation of the ResNet Architecture using PyTorch Library  
    There are few differences between the Actual Paper and this implementation.
                     
    1.  Use Xavier Normal initialization instead of initializing just from a normal distribution.
    2.  "pre-activation" of the weight layers from "Identity Mappings in Deep Residual Networks" has been incorporated. 
             
"""


class ConvWithPreActivation(CNNBaseModel):
    """
        "pre-activation" of the weight layers from "Identity Mappings in Deep Residual Networks" has been incorporated by adding the
        Batch Normalization layer before Convolution layer.

        This class is for defining the following 3 layers together. It extends the CNNBaseModel parent class.
            1. BatchNorm2d
            2. Conv2d
            3. ReLU

        The only purpose of this class is to reduce number of coding lines.
    """

    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        """
            The constructor of the DefaultConvolutionModule class.
            :param in_channels: number of input channel
            :param out_channels: output channel
            :param kernel: size of the kernel
            :param stride: size of stride
            :param padding: size of padding
        """
        super(ConvWithPreActivation, self).__init__()

        self.model = torch.nn.Sequential(
            # Define the BatchNorm2d
            # momentum –  the value used for the running_mean and running_var computation. Default is 0.1.
            torch.nn.BatchNorm2d(num_features=in_channels, momentum=0.9),
            # Define the convolution layer. The output spacial dimension remains same after convolution.
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            # Define ReLU Activation.
            torch.nn.ReLU())

        # Initialize the layers using xavier normal
        self.weights_init_xavier_normal()

    def forward(self, x):
        """
            The forward function for the ConvWithPreActivation

            :param x: input data
        """
        return self.model(x)


class ResNetBottleNeck(CNNBaseModel):
    """
        This class is  responsible for creating the Residual Block with Bottle Neck. It creates 3 ConvWithPreActivation layers.
    """

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(ResNetBottleNeck, self).__init__()

        layers = []

        # 1x1 Convolution Layer with out_channel is 1/4 of actual out_channels
        layers += [ConvWithPreActivation(in_channels=in_channels, out_channels=int(out_channels * 0.25))]

        # 3x3 Convolution Layer with out_channel is 1/4 of actual out_channels
        # stride is always 1 so that no change to spacial dimension
        layers += [
            ConvWithPreActivation(in_channels=int(out_channels * 0.25), out_channels=int(out_channels * 0.25), kernel=3, stride=stride, padding=1)]

        # 1x1 Convolution Layer to increase channel size
        layers += [ConvWithPreActivation(in_channels=int(out_channels * 0.25), out_channels=out_channels)]

        self.model = torch.nn.Sequential(*layers)

        # create a variable named
        self.down_sample = None

        # Need to resize the identity shortcut in 2 scenarios
        #   1. When the first residual block on the repeated sub blocks uses stride of 2 to half the spacial dimension. However this in not applicable
        #      for the very first residual block as the special dimension does not change from 56.
        #      Whenever stride is 2 we need to use this Conv2d layer.
        #   2. For the very first residual block, input channel size is 64 however output channel size is 256. In order to perform the addition
        #      (shortcut) the channel size need to be same.
        #      Hence whenever first_block is True, use this Conv2d layer to match the out channel. Remember the stride is 1 in this case as the
        #      spacial dimension must remain same ( 56 )
        if stride > 1 or first_block:
            # The down_sample can also be called as up_channel as we are using this to increase the channel size too.
            self.down_sample = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
            The forward function for the ResNetBottleNeck. Here the shortcut was used.

            :param x: input data
        """
        identity = x
        output = self.model(x)

        # if the down sample is not None and then pass the identity through the additional convolution layer.
        if self.down_sample:
            identity = self.down_sample(identity)

        # Add the identity layer with the bottle neck layer
        output += identity
        return output


class ResNet(CNNBaseModel):
    """
        This class defines the ResNet Architecture.
    """

    def __init__(self, layer_configs, num_classes=256):
        """
            This constructor is responsible for defining the layers in the architecture.
        """
        super(ResNet, self).__init__()
        layers = []

        # Initial Convolution and Max Pooling Layers
        layers += [ConvWithPreActivation(in_channels=3, out_channels=64, kernel=7, stride=2, padding=3)]
        layers += [torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        in_channels = 64

        # Loop through the defined layer configuration
        for i, (out_channels, times) in enumerate(layer_configs):
            # set stride = 2 only then the loop is not in the first residual block as no need to reduce the special dimension for that.
            if i > 0:
                stride = 2
                first_block = False
            else:
                # set first_block to true to match the # of kernels/filters with the bottle neck layer
                stride = 1
                first_block = True

            # Create the first ResNetBottleNeck layer.
            layers += [ResNetBottleNeck(in_channels=in_channels, out_channels=out_channels, stride=stride, first_block=first_block)]

            # Set the in_channels = out_channels to match the in_channels of the next ResNetBottleNeck layer
            in_channels = out_channels

            # Create remaining ResNetBottleNeck in loop ( all with stride 1 )
            for j in range(times - 1):
                layers += [ResNetBottleNeck(in_channels=in_channels, out_channels=out_channels)]

        # AdaptiveAvgPool2d will shrink the layers automatically. No need to provide the kernel.
        layers += [torch.nn.AdaptiveAvgPool2d((1, 1))]
        # Flatten the layers
        layers += [Flatten()]

        # Define linear layers
        layers += [nn.Linear(in_channels, num_classes)]
        # dim - A dimension along which LogSoftmax will be computed.
        # Since our inout is (N, L), we need to pass 1
        layers += [torch.nn.LogSoftmax(dim=1)]

        # Create the Sequential layer
        self.model = torch.nn.Sequential(*layers)

        # Use xavier normal initializer
        self.weights_init_xavier_normal()

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        """
            The forward function for the ResNet

            :param x: input data
        """
        x = self.model(x)
        return x


def resnet_50(num_classes=256):
    """
        This function defines the ResNet50 architecture with 50 convolution layers

        :param num_classes: number of classes
    """
    layer_configs = [(256, 3), (512, 4), (1024, 6), (2048, 3)]
    return ResNet(layer_configs=layer_configs, num_classes=num_classes)


def resnet_38(num_classes=256):
    """
        This function defines the ResNet38 architecture with 50 convolution layers

        :param num_classes: number of classes
    """
    layer_configs = [(128, 3), (256, 3), (512, 4), (1024, 2)]
    return ResNet(layer_configs=layer_configs, num_classes=num_classes)


def resnet_29(num_classes=256):
    """
        This function defines the ResNet29 architecture with 50 convolution layers

        :param num_classes: number of classes
    """
    layer_configs = [(128, 2), (256, 2), (512, 3), (1024, 2)]
    return ResNet(layer_configs=layer_configs, num_classes=num_classes)


def resnet_26(num_classes=256):
    """
        This function defines the ResNet29 architecture with 50 convolution layers

        :param num_classes: number of classes
    """
    layer_configs = [(128, 2), (256, 2), (512, 2), (1024, 2)]
    return ResNet(layer_configs=layer_configs, num_classes=num_classes)
