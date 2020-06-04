# Implementation of ZFNet using PyTorch
ZFNet is a predecessor of AlexNet Paper as it recommended fine tuning AlexNet by visualizing the Convolution
Network. This implementation is an almost exact replica of the ZFNet paper in PyTorch, however there are many
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

There are few differences between this implementation and original AlexNet paper mostly due to obsolete/outdated concepts.

For more information on the AlexNet implementation please refer the following github repository. This implementation is 
based on the AlexNet implementation.  

[Implementation of AlexNet using PyTorch](../AlexNet)

## Dataset
The ZFNet paper uses ImageNet dataset (like AlexNet), however the authors also published their results on **Caltech256** 
dataset which consists of 256 Categories and around 30K images. Any decent GPU should be able to train using this 
dataset in much lesser time than ImageNet. 

Hence we will compare our result using the **Caltech256** dataset.

In order to use ImagNet instead of Caltech256, please find the below blog post for more details.

[How to prepare imagenet dataset for image classification](http://www.adeveloperdiary.com/data-science/computer-vision/how-to-prepare-imagenet-dataset-for-image-classification/)

Below is the URL of the Caltech256 Dataset.

[Download Caltech 256 Dataset](/http://www.vision.caltech.edu/Image_Datasets/Caltech256/#Details)

### Pre-Processing
The pre-processing steps are same as AlexNet. As ZFNet hasn't recommended any additional improvements. 

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
There are no changes to the data augmentation used in AlexNet. Following Data Augmentations are implemented using the 
albumentations library in the `ZFNet.transformation.py` file.

#### Training Data Augmentation    
1. Horizontal Reflection ( Flip )
2. Random Crop of 227x227

    - The ZFNet paper uses 224x224 random crop like AlexNet, however as per our assumption in AlexNet, 227 is used here instead of 224.
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
Additional changes suggested by the ZFNet paper on top of AlexNet have been incorporated on top my AlexNet implementation.

To recap, following lists the changes between original AlexNet paper and my AlexNet implementation.
1. Use of **Batch Normalization** after the activation layer instead of **Local Response Normalization**. 
   ZFNet paper does not use Batch Normalization as it wasn't published at that time. Study indicates 
   Batch Normalization is more robust than Local Response Normalization.
2. Use **Max Pooling** instead of Average Pooling.
3. Use more Dropout layers ( after MaxPool layers ) to reduce over-fitting.
4. Use **Xavier Normal** initialization instead of initializing just from a normal distribution. 

Here are changes suggested in ZFNet:

1. Update the size of the kernel of the **1st convolution layer** from 11 to 7 and stride from 4 to 2. 
2. Changed the stride from 1 to 2 and padding from 2 to 1 for the **2nd convolution layer**.
3. Update the channels of convolution layer **3,4,5** from 384, 384, 256 to 512, 1024, 512.
    - Update the Batch Norm layers accordingly.
4 The input dimension of the **first Fully Connected** layer will be changed to 6 * 6 * 512 from 6 * 6 * 256.

### Layers 
None : In PyTorch input image of 224 x 224 can be used as PyTorch ignores the fraction.

Modified parameters are in bold.

| **Layer Type** | **Output Size** | **Kernel Size** | **# of Kernels** | **Stride** | **Padding** |
|----------------|-----------------|-----------------|------------------|------------|-------------|
| Input Image    | 227 x 227 x 3   |                 |                  |            |             |
| Conv2d         | 111 x 111 x 96  | **7**               | 96               | **7**          |             |
| ReLU           | 111 x 111 x 96  |                 |                  |            |             |
| BatchNorm2d    | 111 x 111 x 96  |                 |                  |            |             |
| MaxPool2d      | 55 x 55 x 96    | 3               |                  | 2          |             |
| Dropout\*      | 55 x 55 x 96    |                 |                  |            |             |
| Conv2d         | 27 x 27 x 256   | 5               | 256              | **2**          | **1**           |
| ReLU           | 27 x 27 x 256   |                 |                  |            |             |
| BatchNorm2d    | 27 x 27 x 256   |                 |                  |            |             |
| MaxPool2d      | 13 x 13 x 256   | 3               |                  | 2          |             |
| Dropout\*      | 13 x 13 x 256   |                 |                  |            |             |
| Conv2d         | 13 x 13 x 512   | 3               | **512**              |            | 1           |
| ReLU           | 13 x 13 x 512   |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 512   |                 |                  |            |             |
| Conv2d         | 13 x 13 x 1024  | 3               | **1024**             |            | 1           |
| ReLU           | 13 x 13 x 1024  |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 1024  |                 |                  |            |             |
| Conv2d         | 13 x 13 x 512   | 3               | **512**              |            | 1           |
| ReLU           | 13 x 13 x 512   |                 |                  |            |             |
| BatchNorm2d    | 13 x 13 x 512   |                 |                  |            |             |
| MaxPool2d      | 6 x 6 x 512     | 3               |                  | 2          |             |
| Dropout\*      | 6 x 6 x 512     |                 |                  |            |             |
| Flatten\(\)    | 6 x 6 x 512     |                 |                  |            |             |
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

![Image of ZFNet](img/zfnet.png)

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

![Training Plot](img/lr.png)

### Comparison with AlexNet
As shown below, the authors of ZFNet were able to achieve 38.8% Accuracy while training from scratch. The implementation 
here was able to achieve 49.67% on the validation set.   

![Caltech256 Result from paper](img/result.png)

ZFNet error rate (top-5) was 1.7% lower then AlexNet. Here we can see that this implementation has around 3% accuracy 
improvement than the AlexNet implementation. 

| **Architecture** | **epochs** | **Training Loss** | **Validation Accuracy** | **Training Accuracy** | **Learning Rate** |
|:----------------:|:----------:|:-----------------:|:-----------------------:|:---------------------:|:-----------------:|
| AlexNet          | 100        | 0\.0777           | 46\.51%                 | 99\.42%               | 0\.01             |
| ZFNet            | 100        | 0\.0701           | 49\.67%                 | 99\.43%               | 0\.01             |

- The network was trained using single NVIDIA 2080ti and 32Bit Floating Point.
- 100 training epochs took 7    5 minutes to complete.     

## How to run the scripts
### Pre-Processing
- Run the following file:
    - `common.preprocessing.image_dir_preprocessor.py`
    - The properties can be changed at `common.preprocessing.properties.py`. Here is how the configurations are defined.
        ```python      
        # Provide the input preprocessing location
        INPUT_PATH = '/media/4TB/datasets/caltech/256_ObjectCategories'
        # Provide the output location to store the processed images
        OUTPUT_PATH = '/media/4TB/datasets/caltech/processed'
        # Validation split. Range - [ 0.0 - 1.0 ]
        VALIDATION_SPLIT = 0.2
        # Output image dimension. ( height,width )
        OUTPUT_DIM = (256, 256)
        # If RGB mean is needed, set this to True
        RGB_MEAN = True
        # If this is false, then the images will only be resized without preserving the aspect ratio.
        CENTER_CROP = True
        
        
        # Function to provide the logic to parse the class labels from the directory.
        def read_class_labels(path):
            return path.split('/')[-1].split('.')[-1]
        ```
### Training & Testing
- Run the following files:
    - `ZFNet.train.py` 
    - `ZFNet.test.py`
        - The test.py will automatically pickup the last saved checkpoint by training
- The properties can be changed at `ZFNet.properties.py`. Here is how the configurations are defined.
```python
config = dict()
config['PROJECT_NAME'] = 'zfnet'
config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed'

config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"

config['CHECKPOINT_INTERVAL'] = 10
config['NUM_CLASSES'] = 256
config['EPOCHS'] = 100  

config['MULTI_GPU'] = False
config['FP16_MIXED'] = False

config["LOGFILE"] = "output.log"
config["LOGLEVEL"] = "INFO"
```

### Console Output
I am executing the script remotely from pycharm. Here is a sample output of the train.py

```
sudo+ssh://home@192.168.50.106:22/home/home/.virtualenvs/dl4cv/bin/python3 -u /home/home/Documents/synch/mini_projects/ZFNet/train.py
Building model ...
Training starting now ...
100%|██████████| 95/95 [01:33<00:00,  1.02 batches/s, epoch=1, loss=5.4724, val acc=12.741, train acc=7.866, lr=0.01]                                                                                   
100%|██████████| 95/95 [00:36<00:00,  2.58 batches/s, epoch=2, loss=4.6682, val acc=16.612, train acc=14.038, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:35<00:00,  2.64 batches/s, epoch=3, loss=4.1887, val acc=20.304, train acc=18.688, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:36<00:00,  2.62 batches/s, epoch=4, loss=3.8182, val acc=22.836, train acc=23.244, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:36<00:00,  2.61 batches/s, epoch=5, loss=3.4973, val acc=25.694, train acc=27.669, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:36<00:00,  2.61 batches/s, epoch=6, loss=3.2096, val acc=29.353, train acc=31.423, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:36<00:00,  2.59 batches/s, epoch=7, loss=2.9447, val acc=31.477, train acc=35.477, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:37<00:00,  2.56 batches/s, epoch=8, loss=2.7364, val acc=33.192, train acc=38.692, lr=0.01]                                                                                  
100%|██████████| 95/95 [00:36<00:00,  2.61 batches/s, epoch=9, loss=2.5339, val acc=34.335, train acc=42.22, lr=0.01]                                                                                   
100%|██████████| 95/95 [00:36<00:00,  2.62 batches/s, epoch=10, loss=2.3231, val acc=34.025, train acc=45.625, lr=0.01]  
```

     
## References
<a id="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">[1]</a> ImageNet Classification with Deep Convolutional Neural Networks

<a id="https://arxiv.org/abs/1502.03167">[2]</a> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

<a id="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">[3]</a> Understanding the difficulty of training deep feedforward neural networks

![[4] Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf)
 


