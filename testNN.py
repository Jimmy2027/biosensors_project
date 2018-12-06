import data_loader as dl
import os
import sendmail
import bignetwork
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import h5py
from matplotlib import pyplot as plt


def testnetwork(img_shape, kernel_size,nx,ny):

    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # (12,28)
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))  # (6,14)
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Decoder Layers
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # (12,28)
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # (24,56)
    model.add(Dropout(0.25))
    model.add(Conv2D(2, kernel_size, activation='relu', padding='same'))
    print(model.output_shape)

    model.add(Reshape(  2*nx*ny))
    model.add(Activation('sigmoid'))
    print(model.output_shape)

    model.add(Reshape(img_shape))

    # model.add(Flatten())
    # model.add(Dense(2*nx*ny))                       #  2=#labels
    # model.add(Activation("relu"))
    # print(model.output_shape)
    # # # softmax classifier
    # model.add(Dense(nx*ny*2))
    # model.add(Activation("sigmoid"))
    # print(model.output_shape)
    # print(img_shape)
    #
    # model.add(Reshape(img_shape))
    # print(model.output_shape)

    generaltheoryofgravity= [model, 'bignetwork']

    return generaltheoryofgravity
