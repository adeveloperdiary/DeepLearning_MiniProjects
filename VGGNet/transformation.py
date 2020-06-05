import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transformation = A.Compose([
    # A.RandomRotate90(p=0.5),
    # A.HorizontalFlip(p=0.5),
    A.RandomCrop(224, 224, p=1.0),
    ToTensorV2(p=1.0)
])

test_transformation = A.Compose([
    A.Resize(224, 224, p=1.0),
    ToTensorV2(p=1.0)
])
