import tensorflow as tf
import random


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
