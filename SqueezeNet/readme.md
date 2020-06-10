# Implementation of SqueezeNet using PyTorch
This is the implementation of SqueezeNet, however there are many other common factors that were taken care such as:

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

There are few functionality which were not implemented here,such as
1. Simple and Complex Bypass
2. Deep Compression  

This implementation is focused on **Vanilla SqueezeNet**  

## Dataset
The SqueezeNet paper used ImageNet dataset, however this implementation used another dataset named **Caltech256** which is very similar to Imagenet but 
consists of only 256 Categories and around 30K images. Any decent GPU should be able to train using this  dataset in much 
lesser time than ImageNet. 

In order to use ImagNet instead of Caltech256, please find the below blog post for more details.

[How to prepare imagenet dataset for image classification](http://www.adeveloperdiary.com/data-science/computer-vision/how-to-prepare-imagenet-dataset-for-image-classification/)

Below is the URL of the Caltech256 Dataset.

[Download Caltech 256 Dataset](/http://www.vision.caltech.edu/Image_Datasets/Caltech256/#Details)

### Pre-Processing
The pre-processing steps are similar to AlexNet. As SqueezeNet hasn't recommended any additional improvements. 

1. Create Train/Validation Dataset ( Test labels are not given )
2. Resize the smaller side of the image to 256 and scale the larger side accordingly.. 
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
albumentations library in the `SqueezeNet.transformation.py` file.

#### Training Data Augmentation    
1. Random Crop of 224x224    
2. RGB Mean Normalization 
3. ShiftScaleRotate
3. Horizontal Flip    
    
#### Testing Data Augmentation
1. Random Crop of 224x224 ( Same as training )
2. RGB Mean Normalization

## CNN Architecture

Here are some of the changed applied in this implementation.
1.  Use Xavier Normal initialization instead of initializing just from a normal distribution.  
2.  Use of SqueezeNet 1.1 Architecture which reduces the computation by 2.4 times.

Here are the layers defined by the authors.
 
![SqueezeNet Layers](img/squeezenet.png)

The Fire module is defined as:

![SqueezeNet Bottleneck](img/fire.png)
 

### Layers 

Here is the layer structure of SqueezeNet 1.1 architecture. 

| **Layer Type**           | **Output Size**     | **Kernel Size**     | **\# of Kernels**     | **Stride**     | **Padding**     |
|--------------------------|---------------------|---------------------|-----------------------|----------------|-----------------|
| Input Image              | 224 x 224 x 3       |                     |                       |                |                 |
| Conv2d                   | 111 x 111 x 64      | 3                   | 64                    | 2              | 3               |
| ELU                      | 111 x 111 x 64      |                     |                       |                |                 |
| MaxPool2d                | 55 x 55 x 128       | 3                   |                       | 2              |                 |
| **FireModule**           | 55 x 55 x 128       |                     |                       |                |                 |
| **FireModule**           | 55 x 55 x 128       |                     |                       |                |                 |
| MaxPool2d                | 27 x 27 x 128       | 3                   |                       | 2              |                 |
| **FireModule**           | 27 x 27 x 256       |                     |                       |                |                 |
| **FireModule**           | 27 x 27 x 256       |                     |                       |                |                 |
| MaxPool2d                | 13 x 13 x 256       | 3                   |                       | 2              |                 |
| **FireModule**           | 13 x 13 x 384       |                     |                       |                |                 |
| **FireModule**           | 13 x 13 x 384       |                     |                       |                |                 |
| **FireModule**           | 13 x 13 x 512       |                     |                       |                |                 |
| **FireModule**           | 13 x 13 x 512       |                     |                       |                |                 |
| Dropout                  | 13 x 13 x 512       |                     |                       |                |                 |
| Conv2d                   | 13 x 13 x 256       | 1                   | 256                   |                |                 |
| ELU                      | 13 x 13 x 256       |                     |                       |                |                 |
| AdaptiveAvgPool2d        | 1 x 1 x 256         |                     |                       |                |                 |
| Flatten                  | 1 x 256             |                     |                       |                |                 |
| LogSoftmax               | 1 x 256             |                     |                       |                |                 |

## Training & Result
### Training Parameters
- Used **Adam** with **CosineAnnealingLR** learning rate scheduler.      
- Initial **Learning Rate** for Adam has been set to `0.001`
- The initial hyper-parameters of CosineAnnealingLR are set as following:
    - T_max   : 5
    - eta_min : 1e-5 
- After 70 epochs the **eta_min** hyper-parameter of CosineAnnealingLR was changed to 1e-6.

#### Result
Here is the plot of Training/Validation Loss/Accuracy after 90 Epochs. We can get more accuracy by using a larger model or
more advanced optimization technique. 

![Training Plot](img/plot.png)

The is the plot of the learning rate decay.  

![Training Plot](img/lr.png)

## Comparison with other architectures implemented
As shown below, the implemented model was able to achieve 46.60% Accuracy while training from scratch which is similar to 
AlexNet.

| **Architecture** | **epochs** | **Training Loss** | **Validation Accuracy** | **Training Accuracy** |
|:----------------:|:----------:|:-----------------:|:-----------------------:|:---------------------:|
| AlexNet          | 100        | 0\.0777           | 46\.51%                 | 99\.42%               |
| ZFNet            | 100        | 0\.0701           | 49\.67%                 | 99\.43%               |
| SqueezeNet_Adam  | 90         | 0\.8037           | 46\.60%                 | 79\.89%               |


- The network was trained using 2 x NVIDIA 2080ti and 32Bit Floating Point.
- 80 training epochs took ~40 Minutes to complete.     

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
        # If this is true, then the images will only be resized while preserving the aspect ratio.
        CENTER_CROP = False
        # If this is true then the smaller side will be resized to the dimension defined above
        SMALLER_SIDE_RESIZE = True        
        
        # Function to provide the logic to parse the class labels from the directory.
        def read_class_labels(path):
            return path.split('/')[-1].split('.')[-1]
        ```
### Training & Testing
- Run the following files:
    - `SqueezeNet.train.py` 
    - `SqueezeNet.test.py`
        - The test.py will automatically pickup the last saved checkpoint by training
- The properties can be changed at `SqueezeNet.properties.py`. Here is how the configurations are defined.

```python
config = dict()
config['PROJECT_NAME'] = 'SqueezeNet'
config['INPUT_DIR'] = '/media/4TB/datasets/caltech/processed'

config['TRAIN_DIR'] = f"{config['INPUT_DIR']}/train"
config['VALID_DIR'] = f"{config['INPUT_DIR']}/val"

config['TRAIN_CSV'] = f"{config['INPUT_DIR']}/train.csv"
config['VALID_CSV'] = f"{config['INPUT_DIR']}/val.csv"

config['CHECKPOINT_INTERVAL'] = 10
config['NUM_CLASSES'] = 256
config['EPOCHS'] = 80  

config['MULTI_GPU'] = True
config['FP16_MIXED'] = False

config["LOGFILE"] = "output.log"
config["LOGLEVEL"] = "INFO"
```

### Console Output
I am executing the script remotely from pycharm. Here is a sample output of the train.py

```
sudo+ssh://home@192.168.50.106:22/home/home/.virtualenvs/dl4cv/bin/python3 -u /home/home/Documents/synch/mini_projects/SqueezeNet/train.py
Building model ...
Training starting now ...
INFO:root:epoch=1, loss=5.2465, val acc=7.318, train acc=5.263, lr=0.001
INFO:root:epoch=2, loss=4.8779, val acc=9.49, train acc=8.318, lr=0.000905463412215599
INFO:root:epoch=3, loss=4.6193, val acc=11.336, train acc=11.036, lr=0.000657963412215599
INFO:root:epoch=4, loss=4.3906, val acc=14.325, train acc=14.075, lr=0.0003520365877844011
INFO:root:epoch=5, loss=4.1922, val acc=16.335, train acc=16.686, lr=0.00010453658778440106
INFO:root:epoch=6, loss=4.1104, val acc=16.661, train acc=17.85, lr=1e-05
INFO:root:epoch=7, loss=4.1201, val acc=17.168, train acc=17.586, lr=0.00010453658778440102
INFO:root:epoch=8, loss=4.1607, val acc=16.465, train acc=16.76, lr=0.0003520365877844012
INFO:root:epoch=9, loss=4.1984, val acc=16.825, train acc=16.336, lr=0.0006579634122155993
INFO:root:epoch=10, loss=4.1244, val acc=17.364, train acc=17.545, lr=0.0009054634122155996
```

## References
[[1] SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE](https://arxiv.org/pdf/1602.07360.pdf)

[[2] ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[[3] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) 

[[4] Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)


 


