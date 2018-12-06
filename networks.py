from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from matplotlib import pyplot as plt


def biggernetwork(img_shape, kernel_size,Dropout_rate):


    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Decoder Layers
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))

    generaltheoryofgravity = [model, 'biggernetwork']

    return generaltheoryofgravity


def bignetwork(img_shape, kernel_size,Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
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
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))

    generaltheoryofgravity = [model, 'bignetwork']

    return generaltheoryofgravity


def twolayernetwork(img_shape, kernel_size,Dropout_rate):
    model = Sequential()

    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))
    model.add(BatchNormalization())

    generaltheoryofgravity = [model, 'twolayernetwork']

    return generaltheoryofgravity


def semibignetwork(img_shape, kernel_size,Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # (12,28)
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Decoder Layers
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))

    generaltheoryofgravity = [model, 'semibignetwork']

    return generaltheoryofgravity


def smallsegnetwork(img_shape, kernel_size,Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))


    # Decoder Layers
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2, 1, activation='sigmoid', padding='same'))

    generaltheoryofgravity = [model, 'smallsegnetwork']

    return generaltheoryofgravity



def segnetwork(img_shape, kernel_size,Dropout_rate):
    model = Sequential()

    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(Dropout_rate))
    # Decoder Layers
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(Dropout_rate))
    model.add(Conv2D(2, 1, activation='sigmoid', padding='same'))

    generaltheoryofgravity = [model, 'segnetwork']

    return generaltheoryofgravity

