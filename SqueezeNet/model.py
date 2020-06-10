import numpy as np
import cv2

import torch
import torch.nn as nn
from common.utils.model_util import *

"""
    This is the implementation of the SqueezeNet Architecture using PyTorch Library                   
"""


class SqueezeModule(CNNBaseModel):
    """
        This class defines the SqueezeModule where the 1x1 convolution is implemented.
    """

    def __init__(self, in_channels, out_channels):
        """
            This constructor is responsible for defining the layers in the SqueezeModule.
        """
        super(SqueezeModule, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            torch.nn.ELU(inplace=True)
        )
        self.weights_init_xavier_normal()

    def forward(self, x):
        """
            The forward function for the SqueezeModule
            :param x: input data
        """
        return self.model(x)


class ExpandModule(CNNBaseModel):
    """
        This class defines the ExpandModule where the 1x1 and 3x3 convolution is implemented.
    """

    def __init__(self, in_channels, out_channels):
        """
            This constructor is responsible for defining the layers in the ExpandModule.
        """
        super(ExpandModule, self).__init__()
        self.conv_1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            torch.nn.ELU(inplace=True)
        )

        self.conv_3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(inplace=True)
        )

        self.weights_init_xavier_normal()

    def forward(self, x):
        """
            The forward function for the ExpandModule
            :param x: input data
        """
        conv_1x1 = self.conv_1x1(x)
        conv_3x3 = self.conv_3x3(x)

        # Combine the layers in array
        # Concatenate by channel, hence dim=1 (N, C, H, W)
        return torch.cat(tensors=[conv_1x1, conv_3x3], dim=1)


class FireModule(CNNBaseModel):
    """
        This class defines the FireModule of the SqueezeNet architecture.
    """

    def __init__(self, in_channels, squeeze_filter, expand_filter):
        super(FireModule, self).__init__()

        # Define the Squeeze and Expand Module
        self.model = torch.nn.Sequential(
            SqueezeModule(in_channels=in_channels, out_channels=squeeze_filter),
            ExpandModule(in_channels=squeeze_filter, out_channels=expand_filter)
        )

    def forward(self, x):
        """
            The forward function for the FireModule
            :param x: input data
        """
        return self.model(x)


class SqueezeNet(CNNBaseModel):
    """
        This class defines the SqueezeNet Architecture.
    """

    def __init__(self, num_classes=256):
        """
            This constructor is responsible for defining the layers in the architecture.
        """
        super(SqueezeNet, self).__init__()

        self.model = torch.nn.Sequential(

            # Initial Layers
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            # Fire Modules
            FireModule(in_channels=64, squeeze_filter=16, expand_filter=64),
            FireModule(in_channels=64 * 2, squeeze_filter=16, expand_filter=64),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModule(in_channels=64 * 2, squeeze_filter=32, expand_filter=128),
            FireModule(in_channels=128 * 2, squeeze_filter=32, expand_filter=128),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModule(in_channels=128 * 2, squeeze_filter=48, expand_filter=192),
            FireModule(in_channels=192 * 2, squeeze_filter=48, expand_filter=192),
            FireModule(in_channels=192 * 2, squeeze_filter=64, expand_filter=256),
            FireModule(in_channels=256 * 2, squeeze_filter=64, expand_filter=256),

            torch.nn.Dropout(p=0.5),

            # Have number of channels equal to number of classes
            torch.nn.Conv2d(in_channels=256 * 2, out_channels=num_classes, kernel_size=1, stride=1),
            torch.nn.ELU(inplace=True),

            # AdaptiveAvgPool2d will shrink the layers automatically. No need to provide the kernel.
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),

            # Linear Layer is not needed
            torch.nn.LogSoftmax(dim=1)
        )

        # Use xavier normal initializer
        self.weights_init_xavier_normal()

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        """
            The forward function for the SqueezeNet

            :param x: input data
        """
        x = self.model(x)
        return x

m=SqueezeNet()
m.print_network()