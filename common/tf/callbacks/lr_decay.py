import numpy as np
import cv2
import tensorflow as tf


def exponential_lr_decay(lr=0.01, s=20):
    def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1 ** (epoch / s)

        return exponential_decay_fn

    return tf.keras.callbacks.LearningRateScheduler(exponential_decay(lr, s))


def piecewise_constant_decay():
    def piecewise_constant_fn(epoch):
        if epoch < 5:
            return 0.01
        elif epoch < 15:
            return 0.005
        else:
            return 0.001

    return tf.keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
