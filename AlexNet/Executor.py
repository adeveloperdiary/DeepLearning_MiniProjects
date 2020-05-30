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


class BaseExecutor(object):
    DEFAULTS = {
        'CHECKPOINT_PATH': ''
    }

    def __init__(self, data_loaders, config):
        self.__dict__.update(Executor.DEFAULTS, **config)
        self.train_data_loader = data_loaders['TRAIN']
        self.val_data_loader = data_loaders['VAL']
        self.test_data_loader = data_loaders['TEST']


class Executor(BaseExecutor):
    def __init__(self, version, data_loaders, config):
        super().__init__()
        self.version = version


e = Executor("", "", config=config)
