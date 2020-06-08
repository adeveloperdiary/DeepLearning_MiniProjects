# Implementation of ResNet with Identity Mappings using PyTorch
This is the implementation of ResNet with Identity Mappings in PyTorch, however there are many other common factors that were taken care such as:

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
7.  The network layers sizes can be printed to console for verification.  

## Dataset
The ResNet paper used ImageNet dataset, however this implementation used another dataset named **Caltech256** which is very similar to Imagenet but 
consists of only 256 Categories and around 30K images. Any decent GPU should be able to train using this  dataset in much 
lesser time than ImageNet. 

In order to use ImagNet instead of Caltech256, please find the below blog post for more details.

[How to prepare imagenet dataset for image classification](http://www.adeveloperdiary.com/data-science/computer-vision/how-to-prepare-imagenet-dataset-for-image-classification/)

Below is the URL of the Caltech256 Dataset.

[Download Caltech 256 Dataset](/http://www.vision.caltech.edu/Image_Datasets/Caltech256/#Details)

### Pre-Processing
The pre-processing steps are similar to AlexNet. As ResNet hasn't recommended any additional improvements. 

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
There were only few types of data augmentation used. Following Data Augmentations are implemented using the 
albumentations library in the `ResNet.transformation.py` file.

#### Training Data Augmentation    
1. Random Crop of 224x224
    - Original paper used center crop, I will be using Random Crop here.
2. Mean RGB Normalization ( Like AlexNet, ZFNet ) 
3. Horizontal Flip
4. Random 90 Degree Rotation    
    
#### Testing Data Augmentation
1. Random Crop of 224x224 ( Same as training )
2. Mean RGB Normalization. 

## CNN Architecture

Here are some of the changed applied in this implementation.
1.  Use Xavier Normal initialization instead of initializing just from a normal distribution.  

Here are the layers defined by the authors.
 
![ResNet Layers](img/layers.png)

Using the bottleneck design for better performance. Have 3 convolution layers, 1x1 (1/4 channel size) -> 3x3 (1/4 channel size) -> 1x1

![ResNet Bottleneck](img/bottleneck.png)

This implementation uses ResNet with Identity Mappings, hence have the Batch Normalization layer before the convolution layer.

![Identity Mappings](img/identity.png)

Here for the Caltech256 dataset used 29 layers of network with the following setup. 

### Layers 

| **Layer Type**           | **Output Size**     | **Kernel Size**     | **\# of Kernels**     | **Stride**     | **Padding**     |
|--------------------------|---------------------|---------------------|-----------------------|----------------|-----------------|
| Input Image              | 224 x 224 x 3       |                     |                       |                |                 |
| ConvWithPreActivation    | 112 x 112 x 64      | 7                   | 64                    | 2              | 3               |
| MaxPool2d                | 56 x 56 x 64        | 3                   |                       | 2              | 1               |
| **ResNetBottleNeck**         | 56 x 56 x 256       |                     |                       |                |                 |
| **ResNetBottleNeck**         | 56 x 56 x 256       |                     |                       |                |                 |
| **ResNetBottleNeck**         | 56 x 56 x 256       |                     |                       |                |                 |
| ResNetBottleNeck         | 28 x 28 x 512       |                     |                       |                |                 |
| ResNetBottleNeck         | 28 x 28 x 512       |                     |                       |                |                 |
| ResNetBottleNeck         | 28 x 28 x 512       |                     |                       |                |                 |
| ResNetBottleNeck         | 28 x 28 x 512       |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| **ResNetBottleNeck**         | 14 x 14 x 1024      |                     |                       |                |                 |
| ResNetBottleNeck         | 7 x 7 x 2048        |                     |                       |                |                 |
| ResNetBottleNeck         | 7 x 7 x 2048        |                     |                       |                |                 |
| ResNetBottleNeck         | 7 x 7 x 2048        |                     |                       |                |                 |
| ResNetBottleNeck         | 7 x 7 x 2048        |                     |                       |                |                 |
| AdaptiveAvgPool2d        | 1 x 1 x 2048        |                     |                       |                |                 |
| Flatten                  | 1 x 2048            |                     |                       |                |                 |
| Linear                   | 1 x 256             |                     |                       |                |                 |
| LogSoftmax               | 1 x 256             |                     |                       |                |                 |


## Training
- Used **Stochastic Gradient Descent** with **Nesterov's momentum** 
    - Also used **Adam** as alternative approach with initial learning rate as 0.001. 
- Initial **Learning Rate** for SGD has been set to `0.01` ( The authors used 0.001 as initial lr)
- In ResNet the learning rate was reduced manually, however we will be using Learning Rate Scheduler.
  We will use **ReduceLROnPlateau** and reduce the learning rate by a factor of 0.5, if there are no improvements after 3 epochs
    - ReduceLROnPlateau is dependent on the validation set accuracy.  
- Also, used **CosineAnnealingLR** instead of **ReduceLROnPlateau** with **Adam**.

### Graphs
Below is the graph showing the two different types of Identity Mapping used in ResNet.

![Identity Mappings](img/identity_mapping.png) 

## Results

### Approach 1:
Used **Stochastic Gradient Descent** with **Nesterov's momentum** and **ReduceLROnPlateau** Learning rate scheduler.

Here is the plot of Training/Validation Loss/Accuracy after 70 Epochs. The model is clearly over-fitting, more data augmentation will probably help. 

![Training Plot](img/plot.png)

The is the plot of the learning rate decay.  

![Training Plot](img/lr.png)

### Comparison with other architecture
As shown below, the implemented model was able to achieve 55.17% Accuracy while training from scratch.

| **Architecture** | **epochs** | **Training Loss** | **Validation Accuracy** | **Training Accuracy** | **Learning Rate** |
|:----------------:|:----------:|:-----------------:|:-----------------------:|:---------------------:|:-----------------:|
| AlexNet          | 100        | 0\.0777           | 46\.51%                 | 99\.42%               | 0\.01             |
| ZFNet            | 100        | 0\.0701           | 49\.67%                 | 99\.43%               | 0\.01             |
| VGG13            | 70         | 0\.0655           | 53\.45%                 | 99\.08%               | 0\.00125          |
| ResNet_SGD    | 70         | 0\.2786           | 55\.17%                 | 94\.89%               | 1\.953125e-05     |

- The network was trained using single NVIDIA 2080ti and 32Bit Floating Point.
- 70 training epochs took 59.7 Minutes to complete.     

### Approach 2:
Used **Adam** optimizer with **CosineAnnealingLR** Learning rate scheduler. This approach produces better validation
set accuracy than previous one.

Here is the plot of Training/Validation Loss/Accuracy after 90 Epochs. The model is clearly over-fitting, more data augmentation will probably help. 


![Training Plot](img/plot_a.png)

The is the plot of the learning rate decay. For first 70 epochs the leaning rate was set between 1e-03 - 1e-05 and from 70-90 
the learning rate was between 1e-04 - 1e-07.

![Training Plot](img/lr_a.png)

### Comparison with other architecture
As shown below, the implemented model was able to achieve 61.51% Accuracy while training from scratch.

| **Architecture** | **epochs** | **Training Loss** | **Validation Accuracy** | **Training Accuracy** | **Learning Rate**       |
|:----------------:|:----------:|:-----------------:|:-----------------------:|:---------------------:|:-----------------------:|
| AlexNet          | 100        | 0\.0777           | 46\.51%                 | 99\.42%               | 0\.01                   |
| ZFNet            | 100        | 0\.0701           | 49\.67%                 | 99\.43%               | 0\.01                   |
| VGG13            | 70         | 0\.0655           | 53\.45%                 | 99\.08%               | 0\.00125                |
| ResNet_SGD    | 70         | 0\.2786           | 55\.17%                 | 94\.89%               | 1\.953125e-05           |
| ResNet_Adam   | 90         | 0\.3104           | 61\.51%                 | 93\.64%               | 9\.63960113097139e-06   |

- The network was trained using single NVIDIA 2080ti and 32Bit Floating Point.
- 90 training epochs took 84.7 Minutes to complete.     

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
    - `ResNet.train.py` 
    - `ResNet.test.py`
        - The test.py will automatically pickup the last saved checkpoint by training
- The properties can be changed at `ResNet.properties.py`. Here is how the configurations are defined.

```python
config = dict()
config['PROJECT_NAME'] = 'ResNet'
config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed'

config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"

config['CHECKPOINT_INTERVAL'] = 10
config['NUM_CLASSES'] = 256
config['EPOCHS'] = 70  

config['MULTI_GPU'] = False
config['FP16_MIXED'] = False

config["LOGFILE"] = "output.log"
config["LOGLEVEL"] = "INFO"
```

### Console Output
I am executing the script remotely from pycharm. Here is a sample output of the train.py

```
sudo+ssh://home@192.168.50.106:22/home/home/.virtualenvs/dl4cv/bin/python3 -u /home/home/Documents/synch/mini_projects/ResNet/train.py
Building model ...
Training starting now ...
100%|██████████| 191/191 [00:53<00:00,  3.58 batches/s, epoch=1, loss=4.9997, val acc=9.67, train acc=8.201, lr=0.001]                                                                                  
100%|██████████| 191/191 [00:52<00:00,  3.61 batches/s, epoch=2, loss=4.499, val acc=13.639, train acc=12.128, lr=0.0009046039886902864]                                                                
100%|██████████| 191/191 [00:53<00:00,  3.59 batches/s, epoch=3, loss=4.1048, val acc=16.923, train acc=16.091, lr=0.0006548539886902863]                                                               
100%|██████████| 191/191 [00:53<00:00,  3.60 batches/s, epoch=4, loss=3.7222, val acc=21.251, train acc=20.967, lr=0.0003461460113097138]                                                               
100%|██████████| 191/191 [00:53<00:00,  3.59 batches/s, epoch=5, loss=3.4461, val acc=25.368, train acc=25.36, lr=9.639601130971379e-05]                                                                
100%|██████████| 191/191 [00:53<00:00,  3.58 batches/s, epoch=6, loss=3.3391, val acc=26.527, train acc=27.127, lr=1e-06]                                                                                                                                              
100%|██████████| 191/191 [00:53<00:00,  3.58 batches/s, epoch=8, loss=3.4304, val acc=24.567, train acc=25.356, lr=0.000346146011309714]                                                                
100%|██████████| 191/191 [00:53<00:00,  3.59 batches/s, epoch=9, loss=3.4815, val acc=21.66, train acc=24.452, lr=0.0006548539886902867]
100%|██████████| 191/191 [00:53<00:00,  3.56 batches/s, epoch=10, loss=3.4207, val acc=25.025, train acc=25.094, lr=0.000904603988690287]
```

## References
[[1] Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

[[2] Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

[[3] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) 

[[4] Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)


 


