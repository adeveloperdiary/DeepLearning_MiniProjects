import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from model import AlexNetModel
from properties import *
from common.dataset.dataset import ClassificationDataset
import pandas as pd
import matplotlib.pyplot as plt

val_df = pd.read_csv(VALID_CSV)

val_dataset = ClassificationDataset(VALID_DIR, val_df, None, False)

val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

images, ids, _ = next(iter(val_data_loader))
print(ids.shape, images.shape)

image = images[0]
image = torch.Tensor.cpu(image).detach().numpy()
plt.imsave(f'{INPUT_DIR}/test.jpg', image)
