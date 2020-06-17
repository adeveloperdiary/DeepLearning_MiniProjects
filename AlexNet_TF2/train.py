import numpy as np
import cv2
from common.tf.dataset.dataset import *
from AlexNet_TF2.properties import *
from AlexNet_TF2.executor import *
import timeit
from AlexNet_TF2.transformation import *

if __name__ == '__main__':
    fields = {'image': 'image', 'label': 'class'}

    train_dataset = get_dataset_from_tfrecord(train=True, batch_size=config['TRAIN_BATCH_SIZE'], tf_files=f"{config['TRAIN_DIR']}.tfrecord",
                                              train_transformation=tf_train_transformation)
    val_dataset = get_dataset_from_tfrecord(tf_files=f"{config['VALID_DIR']}.tfrecord")

    e = Executor("", {'TRAIN': train_dataset, 'VAL': val_dataset}, config=config)

    start = timeit.default_timer()
    e.train()
    stop = timeit.default_timer()
    print(f'Training Time: {round((stop - start) / 60, 2)} Minutes')
