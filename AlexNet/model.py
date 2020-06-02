import numpy as np
import cv2

import torch
import torch.nn as nn
import common.utils.model_util as model_util


class AlexNetModel(torch.nn.Module):
    def __init__(self, num_classes=256):
        super(AlexNetModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=96),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=256),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=384),

            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=384),

            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=256),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            model_util.Flatten(),

            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(6 * 6 * 256, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

        self.model.apply(model_util.weights_init_xavier_uniform)

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.model(x)
        return x
