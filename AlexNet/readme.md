# Implementation of AlexNet using PyTorch
This implementation is an almost exact replica of the AlexNet paper in PyTorch, however there are many
common factors that were taken care such as:

1.  Data Augmentation is outside of main class and can be defined in a 
    semi declarative way using albumentations library inside the transformation.py class.
2.  Automatic Loading and Saving models from and to checkpoint. 
3.  Integration with Tensor Board. The Tensor Board data is being written after a checkpoint save.
    This is to make sure that, upon restarting the training, the plots are properly drawn.
        A.  Both Training Loss and Validation Accuracy is being written. The code will be modified to 
            also include Training Accuracy and Validation Loss.
        B.  The model is also being stored as graph for visualization.
4.  Logging has been enabled in both console and external file. The external file name can be configured 
    using the configuration in properties.py.
5.  Multi-GPU Training has been enabled using torch.nn.DataParallel() functionality. 
6.  Mixed Precision has been enabled using Nvidia's apex library as the PyTorch 1.6 is not released yet.
    None:   At this moment both Multi-GPU and Mixed Precision can not be using together. This will be fixed 
            once PyTorch 1.6 has been released. 

There are few differences between this implementation and original paper mostly due to obsolete/outdated concepts.
Each section will elaborate difference in detail along with additional explanations. 

## Dataset
The AlexNet paper uses ImageNet dataset, however here we will be using Caltech256 dataset which consists of 256 
Categories and around 30K images. Any decent GPU should be able to train using this dataset in much lesser time than 
ImageNet.

In order to use ImagNet instead of Caltech256, please find the below blog post for more details.

http://www.adeveloperdiary.com/data-science/computer-vision/how-to-prepare-imagenet-dataset-for-image-classification/

Below is the URL of the Caltech256 Dataset.

http://www.vision.caltech.edu/Image_Datasets/Caltech256/#Details

### Pre-Processing
The pre-processing steps are same as AlexNet. Here are the steps:

1. Create Train/Validation Dataset ( Test labels are not given )
2. Center crop images 
3. Resize image to 256x256 Pixels
4. Calculate RGB Mean ( only on train set ) and finally save the global mean to a file named `rgb_val.json`.
    - The RGB mean values is used during training to normalize each images in `ClassificationDataset` class.
5. Moves the processed images to a different dir
6. Create a file name `categories.csv` with the list if class labels and corresponding ids.
7. Create train/val csv file with image name ( randomly generated ) and class id.

The `common.preprocessing.image_dir_preprocessor.py` class performs the pre processing tasks. 

None: In case of ImageNet, parallel processing is recommended. Please refer the below blog post for more details.

http://www.adeveloperdiary.com/data-science/computer-vision/imagenet-preprocessing-using-tfrecord-and-tensorflow-2-0-data-api/

### Data Augmentation
Following Data Augmentations are implemented using the albumentations library in the `AlexNet.transformation.py` file.

#### Training Data Augmentation    
1. Horizontal Reflection ( Flip )
2. Random Crop of 227x227
    - The AlexNet paper uses 224x224 random crop, however many believe the actual value is 227 instead of 224.
    - Also AlexNet uses 5 Crops 1 Center Crop and 4 sides crop, total 10 crops per images. However here we will
      Just use RandomCrop() feature of albumentations library.        
3.  PCA Color Augmentation
    - Even though the AlexNet paper uses PCA Color Augmentation, this PyTorch implementation does not use that, as
      the batch normalization is powerful  to cancel the effect of PCA Color Augmentation. Please refer the github 
      project for more information.
      
      https://github.com/koshian2/PCAColorAugmentation
      
    
#### Testing Data Augmentation
1. Random Crop of 227x227 ( Same as training )        





