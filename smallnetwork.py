from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from matplotlib import pyplot as plt
import numpy as np


def smallnetwork(img_shape, kernel_size):
    model = Sequential()

    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))
    model.add(BatchNormalization())

    generaltheoryofgravity= [model, 'smallnetwork']

    return generaltheoryofgravity

