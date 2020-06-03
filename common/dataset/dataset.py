import numpy as np
import cv2
import torch
import pandas as pd
from sklearn.utils import shuffle
import albumentations as A
import json


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, data_frame, transform, fields={}, training=True, mean_rgb=None):
        super().__init__()
        self.image_dir = image_dir
        self.data_frame = data_frame
        self.transform = transform
        self.training = training
        self.fields = fields
        self.image_ids = data_frame[self.fields['image']].unique()
        self.image_ids = shuffle(self.image_ids)
        self.rgb_means = None
        if mean_rgb:
            self.rgb_means = json.loads(open(mean_rgb, 'r').read())

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_class = self.data_frame[self.data_frame[self.fields['image']] == image_id][self.fields['label']]
        image_class = image_class.values

        # Convert the label to torch int64
        image_class = torch.as_tensor(image_class, dtype=torch.int64)

        # Read the image from disk using open cv
        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Create the dict needed for transformation
            transform_input = {
                'image': image
            }
            # Transform the image
            transform_output = self.transform(**transform_input)

            # Get the transformed image
            image = transform_output['image']

        if self.rgb_means:
            # Split the image to separate channel.
            # Since we have already converted the image from BGR to RGB
            # we will get the channel as RGB here.
            (R, G, B) = cv2.split(image.astype('float32'))

            # Subtract the mean
            R -= self.rgb_means['R']
            G -= self.rgb_means['G']
            B -= self.rgb_means['B']

            # Merge the channels
            image = cv2.merge([R, G, B])
            # Convert the image to PyTorch Tensor
            image = torch.as_tensor(image, dtype=torch.float32)
        else:
            # Convert the image to PyTorch Tensor
            image = torch.as_tensor(image, dtype=torch.float32)
            # Basic Normalization by dividing 255.0
            image /= 255.0

        return image, image_class, image_id
