import numpy as np
import cv2

import torch


class AlexNetModel(torch.nn.Module):
    def __init__(self, num_classes=256):
        super(AlexNetModel, self).__init__()
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=96),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=256),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=384),

            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=384),

            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=256),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Dropout(),

            torch.nn.Linear(6 * 6 * 256, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
