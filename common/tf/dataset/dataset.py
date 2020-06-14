import numpy as np
import cv2
import tensorflow as tf
import random
import pandas as pd


def read_from_tfrecord(size):
    def inner_tfrecord(record):
        features = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(record, features)
        image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
        label = tf.cast(parsed_record['label'], tf.int32)
        image = tf.image.random_crop(image, size=[size, size, 3])
        return image, label

    return inner_tfrecord


def read_from_fs(size):
    def inner_fs(image_path, label):
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.random_crop(image, [size, size, 3])
        label = tf.cast(label, tf.int32)

        return image, label

    return inner_fs


def tf_train_transformation(image, label):
    image = tf.image.random_flip_left_right(image)
    option = random.randint(0, 1)
    if option == 0:
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_brightness(image, 0.4)
    else:
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_contrast(image, 0.7, 1.3)
    return image, label


def convert_to_float32(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def get_dataset_from_tfrecord(image_size=227, batch_size=32, buffer=1000, repeat=-1, tf_files=None, train=False):
    record_files = tf.data.Dataset.list_files(tf_files)
    dataset = tf.data.TFRecordDataset(filenames=record_files)

    dataset = dataset.map(read_from_tfrecord(image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return pre_process(dataset, batch_size=batch_size, buffer=buffer, repeat=repeat, train=train)


def pre_process(dataset, batch_size, buffer, repeat, train):
    if train:
        dataset = dataset.map(tf_train_transformation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(convert_to_float32, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(count=repeat)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer, reshuffle_each_iteration=True)

    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_dataset_from_csv(csv_path, images_path, fields, image_size=227, batch_size=32, buffer=1000, train=False):
    df = pd.read_csv(csv_path)
    paths = df[fields['image']].values
    labels = df[fields['label']].values

    offset = 5
    paths = paths.astype(f'S{len(paths[0]) + offset}')

    image_root = np.empty(paths.shape[0], dtype=f'S{len(images_path)}')
    image_root[:] = images_path
    image_root = image_root.reshape(-1, 1)

    paths = paths.reshape(-1, 1)

    image_paths = np.concatenate((image_root, paths), axis=1)
    image_full_paths = np.apply_along_axis(lambda d: d[0] + d[1], 1, image_paths)

    paths = tf.data.Dataset.from_tensor_slices(image_full_paths)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((paths, labels))

    dataset = dataset.map(read_from_fs(image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return pre_process(dataset, batch_size=batch_size, buffer=buffer, train=train)
