from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten




def biggernetwork(img_shape, kernel_size):

    model = Sequential()
    # Encoder Layers
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same', input_shape=img_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))
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
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(2, kernel_size, activation='sigmoid', padding='same'))


    generaltheoryofgravity= [model, 'biggernetwork']

    return generaltheoryofgravity