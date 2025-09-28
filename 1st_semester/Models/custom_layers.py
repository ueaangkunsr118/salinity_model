import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

class CustomNormalization(tf.keras.layers.Layer):
    def __init__(self, median, iqr, mean, std, epsilon=1e-7, **kwargs):
        super(CustomNormalization, self).__init__(**kwargs)
        self.median = tf.constant(median, dtype=tf.float32)
        self.iqr = tf.constant(iqr, dtype=tf.float32)
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)
        self.epsilon = epsilon

    def call(self, inputs):
        # Apply Robust Scaling: (x - median) / IQR
        x = (inputs - self.median) / (self.iqr + self.epsilon)
        
        # Apply log1p transformation
        x = tf.math.log1p(x)
        
        # Standardize using the mean and std from training
        x = (x - self.mean) / (self.std + self.epsilon)
        
        return x

    def get_config(self):
        config = super(CustomNormalization, self).get_config()
        config.update({
            "median": self.median.numpy().tolist(),
            "iqr": self.iqr.numpy().tolist(),
            "mean": self.mean.numpy().tolist(),
            "std": self.std.numpy().tolist(),
            "epsilon": self.epsilon,
        })
        return config