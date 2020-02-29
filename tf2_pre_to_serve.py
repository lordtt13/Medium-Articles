# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:55:05 2020

@author: Tanmay Thakur
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)