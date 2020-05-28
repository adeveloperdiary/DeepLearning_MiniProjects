import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from model import AlexNetModel
from properties import *
from common.dataset.dataset import ClassificationDataset
import pandas as pd
import matplotlib.pyplot as plt
from common.utils.logging_util import *
from common.utils.training_util import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

# images, ids, _ = next(iter(val_data_loader))
# print(ids.shape, images.shape)

# image = images[0]
# image = torch.Tensor.cpu(image).detach().numpy()
# plt.imsave(f'{INPUT_DIR}/test.jpg', image)

# init_logging()
# create_checkpoint_folder()

transformation = A.Compose([
    A.Resize(227, 227, p=1.0),
    ToTensorV2(p=1.0)
])

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VALID_CSV)

train_dataset = ClassificationDataset(TRAIN_DIR, train_df, transformation, True)
val_dataset = ClassificationDataset(VALID_DIR, val_df, transformation, False)

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

model = AlexNetModel(num_classes=256)
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0005)
criterion = torch.nn.CrossEntropyLoss()
model.cuda()


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


loss_hist = Averager()

for epoch in range(5):
    loss_hist.reset()

    for i, (images, image_class, image_id) in enumerate(train_data_loader):
        images = images.to(DEVICE)
        image_class = image_class.to(DEVICE)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, image_class.squeeze())
        loss.backward()
        optimizer.step()

        loss_hist.send(loss.item())
        if i % 100 == 0:  # print every 2000 mini-batches
            print(f"Epoch #{epoch} Iteration #{i} Average loss: {loss_hist.value}")
