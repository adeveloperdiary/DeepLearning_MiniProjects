import numpy as np
import cv2
import torch
import pandas as pd
from sklearn.utils import shuffle
import albumentations as A


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, data_frame, transform, fields={}, training=True):
        super().__init__()
        self.image_dir = image_dir
        self.data_frame = data_frame
        self.transform = transform
        self.training = training
        self.fields = fields
        self.image_ids = data_frame[self.fields['image']].unique()
        self.image_ids = shuffle(self.image_ids)

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_class = self.data_frame[self.data_frame[self.fields['image']] == image_id][self.fields['label']]
        image_class = image_class.values

        image_class = torch.as_tensor(image_class, dtype=torch.int64)

        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transform_input = {
                'image': image
            }

            transform_output = self.transform(**transform_input)
            image = transform_output['image']

        image = torch.as_tensor(image, dtype=torch.float32)
        image /= 255.0

        return image, image_class, image_id
