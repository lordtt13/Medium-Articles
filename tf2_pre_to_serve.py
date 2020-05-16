# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:55:05 2020

@author: Tanmay Thakur
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)


# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

# number of classes
K = len(set(y_train))
print("number of classes:", K)

def create_model():
  i = Input(shape = x_train[0].shape)

  x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(i)
  x = BatchNormalization()(x)
  x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2))(x)

  x = Flatten()(x)
  x = Dropout(0.2)(x)
  x = Dense(1024, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(K, activation = 'softmax')(x)

  model = Model(i, x)
  return model

strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.CentralStorageStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
  model = create_model()

  model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
  
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5)