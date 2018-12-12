#!usr/bin/env python
"This code trains the NN"

import sendmail
import networks
import testNN
import data_loader as dl
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import h5py
from matplotlib import pyplot as plt

"""
resolution of images
"""
# nx = 24
# ny = 56
# nx = 240
# ny = 560
nx = 600
ny = 1400


# main_dir = '/home/klugh/Documents/Python'         #etz computer
# main_dir = '/cluster/home/klugh/Python'           #eth cluster
# main_dir = '/home/klug/Hendrik/sensors_project/Sensors_project'         #epfl server
# save_dir = '/Users/Hendrik/Desktop/'  # use this for local
# save_dir = main_dir                             #use this for remote
label_dir = 'labeled_images'
training_dir = 'training_images'
test_dir = 'test_images'


"""
x_train: original pictures
y_train: labeled pictures corresponding to x_train
x_test:  original pictures without corresponding labeled pictures (used to create a visual evaluation of Network with plotter)
x_train/y_train_r,g are the red and green channels of the rgb pictures (blue parts are not needed)
"""
x_train = dl.load_images(training_dir, nx, ny)
x_train0 = x_train
x_train = dl.x_preprocessing(x_train)
y_train = dl.load_images(label_dir, nx, ny)
y_train = dl.y_preprocessing(y_train)
x_test = dl.load_images(test_dir, nx, ny)
x_test = dl.x_preprocessing(x_test)

num_of_imageparts = 10                          # number of (256*256) image parts per original picture

"""
Create 10 bits for each of the 33 original x_train and y_train images
"""
xtrain_bits, ytrain_bits = dl.create_random_imagepart(x_train, y_train, num_of_imageparts)



img_shape = (256, 256, 2)

"""
variables of Network: batch_Size, kernel_size, Dropout, weights, validation_split, epochs
"""
Dropout_rate = 0.5
validation_split_val = 0.15
batch_size = 32
epochs = 200


for i in (2, 3):
    kernel_size = i

    networkreturn = networks.segnetwork(img_shape, kernel_size, Dropout_rate)
    model = networkreturn[0]
    whichmodel = networkreturn[1]
    model.summary()

    save_dir = 'models_part/validation_split_' + str(validation_split_val) + '/' + whichmodel + '/' + str(
        epochs) + '_epochs/' + 'Kernel=' + str(kernel_size)
    print(save_dir)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = model.fit(xtrain_bits, ytrain_bits, validation_split=0.25, batch_size=batch_size, nb_epoch=epochs, verbose=1)



    print(history.history.keys())

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_dir, whichmodel + str(kernel_size) + 'accuracy_values.png'))

    plt.figure()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(save_dir, whichmodel + str(kernel_size) + 'loss_values.png'))





    model_save_dir = os.path.join(save_dir,
                                  whichmodel + str(kernel_size) + '(' + str(ny) + '*' + str(nx) + ').h5')



    model.save(model_save_dir)

    # sendmail.sendmail(kernel_size, save_dir, whichmodel+str(kernel_size)+'('+str(ny)+'*'+str(nx)+').h5', whichmodel+str(kernel_size)+'accuracy_values.png',whichmodel+str(kernel_size)+'loss_values.png')

# TODO set small weights for black pixel, so that accuracy is low when network return all black
# TODO make organised data saving dir (sql?)
# TODO setup pw for smtp
# TODO implement keras data augmentation
