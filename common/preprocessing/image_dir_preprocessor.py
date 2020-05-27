import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import tracemalloc
import matplotlib.pyplot as plt
# Change the import as needed
from properties import *
# from .properties import *
import pandas as pd
from sklearn.model_selection import train_test_split
import uuid
import os
import shutil
import json

"""
    This code is used to pre process the Caltech 256 Classification Dataset.
    https://authors.library.caltech.edu/7694/
    
    This script should be used for one time purpose only.
    
    The individual classes are inside a directory where the name of the class 
    is given as <SomeNumber>.<ClassLabel>.
    
    Here are the steps that this code performs:
        1. Create Train/Validation Dataset
        2. Center crop ( optional ) images
        3. Resize image 
        4. Calculate RGB Mean ( only on train set )
        5. Moves the processed images to a different dir
        6. Create a file with the list if class labels and corresponding ids.
        7 Create train/val csv file with image name ( randomly generated ) and class id.
        
    Properties can be set in the properties.py file.        

"""


def resize_image(image):
    """
        Resize and Center Crop the images. Center Crop preserves the aspect ratio.
        However Center Crop might remove additional pixel which can impact the prediction score.
        Advised to perform experimentation before setting Center Crop to True.

        :arguments:
        ----------------------------------------
            image[ndarray] : Binary Image
        :return:
         ---------------------------------------
            Processed binary image.

    """
    if CENTER_CROP:
        # Capture the width and height from the array
        height, width, _ = image.shape
        # Compare width and height. +1 is added to that divide by 2 does not produce 0 values.
        if width > height + 1:
            # Need to crop width
            delta = (width - height) // 2

            # Center Crop
            image = image[:, delta:-delta, :]
        elif height > width + 1:
            # Need to crop height
            delta = (height - width) // 2
            image = image[delta:-delta, :, :]
    image = cv2.resize(image, OUTPUT_DIM)
    return image


def create_dataset(X, y, type):
    """
        This method is responsible to preprocess the images and move to them to a different folder.

        :arguments:
        ----------------------------------------
            X[list]      : The list of image file paths.
            y[list]      : The list of class ids of each image.
            type[string] : Used for creating the train/val folder. Valid values are train/val.
        :return:
         ---------------------------------------
            None
    """
    # Delete the dir if already present
    if os.path.exists(f'{OUTPUT_PATH}/{type}'):
        shutil.rmtree(f'{OUTPUT_PATH}/{type}')

    # Create the dir
    os.mkdir(f'{OUTPUT_PATH}/{type}')

    pbar = tqdm(total=len(X))

    dataset = []

    # Placeholder for RGB Mean Calculation
    (R, G, B) = ([], [], [])

    # Loop through the file paths.
    for i in range(len(X)):
        # Read the image using opencv library.
        image = cv2.imread(X[i])

        # Resize the image
        image = resize_image(image)

        if RGB_MEAN:
            # Calculate the mean RGB for each image.
            # Remember opencv uses BGR format instead of RGB format.
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # Generate unique name for each image.
        name = f'{uuid.uuid4()}.jpg'

        # Save the processed image to a the output folder
        cv2.imwrite(f'{OUTPUT_PATH}/{type}/{name}', image)

        # add the image name and class id to the array
        dataset.append({
            'image': name,
            'class': y[i]
        })

        pbar.update(1)
    pbar.close()

    # export the list as csv
    df = pd.DataFrame(dataset, columns=['image', 'class'])
    df.to_csv(f'{OUTPUT_PATH}/{type}.csv', index=False)

    if RGB_MEAN:
        # Save the mean RGB data in json file if RGB mean calculation has been enabled.
        with open(f'{OUTPUT_PATH}/rgb_{type}.json', "w+") as f:
            f.write(json.dumps({"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}))


def define_train_valid_split():
    """
        Finds all the images and splits them in train and validation set.
        This function uses sklearn's train_test_split function with stratify
        so that the validation set gets images from all classes equality.

        :arguments:
        ----------------------------------------
            None
        :return:
         ---------------------------------------
            The list of train & validation list along with the Label Ids and name dict.
    """

    # Find all the directories
    dirs = glob(f'{INPUT_PATH}/*')
    categories = {}

    # Create a dict with id and class label names.
    for dir_names in dirs:
        categories[read_class_labels(dir_names)] = len(categories)

    # Find all the jpg files
    files = glob(f'{INPUT_PATH}/**/**.jpg')
    paths = []
    labels = []

    pbar = tqdm(total=len(files))
    # Loop through the files and add them to the paths and labels list.
    for i, file in enumerate(files):
        category = file.split('/')[-2].split('.')[-1]
        paths.append(file),
        labels.append(categories[category])

        pbar.update(1)
    pbar.close()

    # Use train_test_split function to create train/validation split.
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=VALIDATION_SPLIT, stratify=labels)

    # switch the key and value of the categories dict
    categories = {categories[k]: k for k in categories}

    return X_train, X_test, y_train, y_test, categories


def process():
    """
        Invokes utility function to perform each operation. Stores the categories dict as a csv file.
    """

    print('Creating Train/Validation Split ... ')
    X_train, X_test, y_train, y_test, categories = define_train_valid_split()

    # Store the categories dict in a csv file.
    cat = []

    for k in categories:
        cat.append({
            'id': k,
            'label': categories[k]
        })

    df = pd.DataFrame(cat, columns=['id', 'label'])
    df.to_csv(f'{OUTPUT_PATH}/categories.csv', index=False)

    print('Creating Train Dataset ... ')
    create_dataset(X_train, y_train, 'train')

    print('Creating Validation Dataset ... ')
    create_dataset(X_test, y_test, 'val')


if __name__ == '__main__':
    """
        Main function to execute the script. This function also prints memory uses.
        If needed multiple parts can be executed in parallel. 
    """
    tracemalloc.start()
    process()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
