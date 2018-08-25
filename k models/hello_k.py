"""
Experimental sequential model using Keras
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(
    Dense(32, input_shape=(784, ))
    )

