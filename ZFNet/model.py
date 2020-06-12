import torch
import common.torch.utils.model_util as model_util

"""
    This is the implementation of the ZFNet Architecture using PyTorch Library  
    There are few differences between the Actual Paper and this implementation.
        
    1.  Use of Batch Normalization after the activation layer instead of Local Response Normalization. 
        ZFNet paper does not use Batch Normalization as it wasn't published at that time. Study indicates 
        Batch Normalization is more robust than Local Response Normalization.
    2.  Use Max Pooling instead of Average Pooling.
    3.  Use more Dropout layers ( after MaxPool layers ) to reduce over-fitting.
    4.  Use Xavier Normal initialization instead of initializing just from a normal distribution.         
          
"""


class ZFNetModel(torch.nn.Module):
    def __init__(self, num_classes=256):
        super(ZFNetModel, self).__init__()

        self.model = torch.nn.Sequential(

            # Define the Input/Output Channel Size, Kernel Size and Stride
            # Changed filter size from 11 -> 7, stride from 4 -> 2
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),

            # inplace=True means that it will modify the input directly,
            # without allocating any additional output. It can sometimes
            # slightly decrease the memory usage, but may not always be a valid operation.
            torch.nn.ReLU(inplace=True),

            # num_features is C from an expected input of size (N, C, H, W)
            torch.nn.BatchNorm2d(num_features=96),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # Additional Dropout Layer
            torch.nn.Dropout(p=0.25),

            # Changed the stride from 1 -> 2 and padding from 2 -> 1
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=256),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # Additional Dropout Layer
            torch.nn.Dropout(p=0.25),

            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=512),

            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=1024),

            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(num_features=512),

            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # Additional Dropout Layer
            torch.nn.Dropout(p=0.25),

            # This is to flatten the input from (N, C, H, W) -> (N, L)
            model_util.Flatten(),

            torch.nn.Linear(6 * 6 * 512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=4096),

            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=4096),

            torch.nn.Linear(4096, num_classes),
            # dim - A dimension along which LogSoftmax will be computed.
            # Since our inout is (N, L), we need to pass 1
            torch.nn.LogSoftmax(dim=1)
        )

        self.model.apply(model_util.weights_init_xavier_normal)

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.model(x)
        return x
