import numpy as np
import cv2
import tensorflow as tf


class CustomCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, save_fn):
        self.save_fn = save_fn

    def on_epoch_end(self, epoch, logs=None):
        self.save_fn(epoch, logs)
