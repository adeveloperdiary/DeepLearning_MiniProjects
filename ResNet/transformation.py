import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transformation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=(-15, -15), border_mode=1, p=0.5),
    A.OneOf([A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=True),
             A.Equalize(p=0.5, mode='cv', by_channels=True), ], p=0.5),
    A.RandomCrop(224, 224, p=1.0),
    ToTensorV2(p=1.0)
])

test_transformation = A.Compose([
    A.Resize(224, 224, p=1.0),
    ToTensorV2(p=1.0)
])
