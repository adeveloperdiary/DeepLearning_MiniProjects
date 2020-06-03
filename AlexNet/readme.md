# Implementation of AlexNet using PyTorch
This implementation is an almost exact replica of the AlexNet paper in PyTorch, however there are many
common factors that were taken care such as:

1.  Data Augmentation is outside of main class and can be defined in a 
    semi declarative way using albumentations library inside the transformation.py class.
2.  Automatic Loading and Saving models from and to **checkpoint**. 
3.  Integration with **Tensor Board**. The Tensor Board data is being written after a checkpoint save.
    This is to make sure that, upon restarting the training, the plots are properly drawn.
        A.  Both Training Loss and Validation Accuracy is being written. The code will be modified to 
            also include Training Accuracy and Validation Loss.
        B.  The model is also being stored as graph for visualization.
4.  **Logging** has been enabled in both console and external file. The external file name can be configured 
    using the configuration in properties.py.
5.  **Multi-GPU Training** has been enabled using `torch.nn.DataParallel()` function. 
6.  **Mixed Precision** has been enabled using Nvidia's apex library as the PyTorch 1.6 is not released yet.
    None:   At this moment both Multi-GPU and Mixed Precision can not be using together. This will be fixed 
            once PyTorch 1.6 has been released. 

There are few differences between this implementation and original paper mostly due to obsolete/outdated concepts.
Each section will elaborate difference in detail along with additional explanations. 

## Dataset
The AlexNet paper uses ImageNet dataset, however here we will be using **Caltech256** dataset which consists of 256 
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
    - Also AlexNet uses 5 Crops ( 1 Center Crop and 4 sides crop), hence total 10 crops per images. However here we will
      Just use RandomCrop() feature of albumentations library. The effect should be very similar. 
             
3.  PCA Color Augmentation
    - Even though the AlexNet paper uses PCA Color Augmentation, this PyTorch implementation does not use that, as
      the batch normalization is powerful  to cancel the effect of PCA Color Augmentation. Please refer the github 
      project for more information.
      
      https://github.com/koshian2/PCAColorAugmentation
      
    
#### Testing Data Augmentation
1. Random Crop of 227x227 ( Same as training )
2. Mean RGB Normalization. 

## CNN Architecture
There are few differences in the CNN Model Architecture between this implementation and the AlexNet paper:

1. Use of **Batch Normalization** after the activation layer instead of **Local Response Normalization**. 
   AlexNet paper does not use Batch Normalization as it wasn't published at that time. Study indicates 
   Batch Normalization is more robust than Local Response Normalization.
2. Use **Max Pooling** instead of Average Pooling.
3. Use more Dropout layers ( after MaxPool layers ) to reduce over-fitting.
4. Use **Xavier Normal** initialization instead of initializing just from a normal distribution. 
   The He paper also refers the AlexNet paper with the following text:
   
 > Recent deep CNNs are mostly initialized by random weights drawn from Gaussian distributions

### Layers 
In PyTorch input image of 224 x 224 can be used as PyTorch ignores the fraction.

| **Layer Type** | **Output Size** | **Kernel Size** | **# of Kernels** | **Stride** | **Padding** |
|----------------|-----------------|-----------------|------------------|------------|-------------|
| Input Image    | 227 x 227 x 3   |                 |                  |            |             |
| Conv2d         | 57 x 57 x 96    | 11              | 96               | 4          |             |
| ReLU           | 57 x 57 x 96    |                 |                  |            |             |
| BatchNorm2d    | 57 x 57 x 96    |                 |                  |            |             |
| MaxPool2d      | 28 x 28 x 96    | 3               |                  | 2          |             |
| Dropout\*      | 28 x 28 x 96    |                 |                  |            |             |
| Conv2d         | 28 x 28 x 256   | 5               | 256              |            | 2           |
| ReLU           | 28 x 28 x 256   |                 |                  |            |             |
| BatchNorm2d    | 28 x 28 x 256   |                 |                  |            |             |
| MaxPool2d      | 13 x 13 x 256   | 3               |                  | 2          |             |
| Dropout\*      | 13 x 13 x 256   |                 |                  |            |             |
| Conv2d         | 13 x 13 x 384   | 3               | 384              |            | 1           |
| ReLU           | 13 x 13 x 384   |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 384   |                 |                  |            |             |
| Conv2d         | 13 x 13 x 384   | 3               | 384              |            | 1           |
| ReLU           | 13 x 13 x 384   |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 384   |                 |                  |            |             |
| Conv2d         | 13 x 13 x 256   | 3               | 256              |            | 1           |
| ReLU           | 13 x 13 x 256   |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 256   |                 |                  |            |             |
| MaxPool2d      | 6 x 6 x 256     | 3               |                  | 2          |             |
| Dropout\*      | 6 x 6 x 256     |                 |                  |            |             |
| Flatten\(\)    | 6 x 6 x 256     |                 |                  |            |             |
| Linear         | 4096            |                 |                  |            |             |
| ReLU           | 4096            |                 |                  |            |             |
| BatchNorm2d    | 4096            |                 |                  |            |             |
| Dropout        | 4096            |                 |                  |            |             |
| Linear         | 4096            |                 |                  |            |             |
| ReLU           | 4096            |                 |                  |            |             |
| BatchNorm2d    | 4096            |                 |                  |            |             |
| Dropout        | 4096            |                 |                  |            |             |
| Linear         | 256             |                 |                  |            |             |
| LogSoftmax     | 256             |                 |                  |            |             |

### Architecture Diagram
Here is the original architecture diagram from the paper.

![Image of AlexNet](img/alexnet.jpeg)

## Training
- Used **Stochastic Gradient Descent** with **Nesterov's momentum** 
- Initial **Learning Rate** has been set to `0.01`
- In AlexNet the learning rate was reduced manually 3 times, by a factor of 10 ( 0.01 -> 0.001 -> 0.0001 -> 0.00001).
  However here we will use **ReduceLROnPlateau** and reduce the learning rate by a factor of 0.5, if there are no improvements after 5 epochs

## Results
Here is the plot of Training/Validation Loss/Accuracy after 100 Epochs. The model is clearly over-fitting, 
more data augmentation will probably help. 

![Training Plot](img/plot.png)

Even though `ReduceLROnPlateau` scheduler was used to decay learning rate, it wasn't effective as the training error kept reducing.
The scheduler started reducing the lr after around 170 epochs to 0.0003125 (Not shown in the plot). 

![Training Plot](img/lr.png)
    

| **epochs**             | **Training Loss** | **Validation Accuracy** | **Training Accuracy** | **Learning Rate** |
|:----------------------:|:-----------------:|:-----------------------:|:---------------------:|:-----------------:|
| 100                    | 0\.0777           | 46\.5%                  | 99\.4%                | 0\.01             |
| 200 \( not in chart \) | 0\.0488           | 59\.3%                  | 99\.6%                | 0\.0003125        |

- The network was trained using single NVIDIA 2080ti and 32Bit Floating Point.
- 200 epochs took 60 minutes to train.     

## References
<a id="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">[1]</a> ImageNet Classification with Deep Convolutional Neural Networks

<a id="https://arxiv.org/abs/1502.03167">[2]</a> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

<a id="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">[3]</a> Understanding the difficulty of training deep feedforward neural networks


