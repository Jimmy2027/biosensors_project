#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:27:54 2018

@author: Hendrik
"""
import x_preprocessing as xprep
import y_preprocessing as yprep
import data_loader as dl
import os
from keras.models import Sequential #keras model module
from keras.layers import Dense, Dropout, Activation, Flatten    #keras core layers
from keras.layers import Convolution3D, MaxPooling3D    #keras CNN layers
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
import numpy as np
from keras.optimizers import Adadelta, RMSprop, adam

#/Users/Hendrik/Documents/Bonus\ programming\ exercise-20181101
# directory where the images are located
main_dir = '/Users/Hendrik/Documents/Bonus programming exercise-20181101'

label_dir = os.path.join(main_dir, 'labeled_images')
training_dir = os.path.join(main_dir, 'training_images')
test_dir = os.path.join(main_dir, 'test_images')



x_train = dl.load_images(training_dir)
x_train0 = x_train
x_train = xprep.x_preprocessing(x_train)
y_train = dl.load_images(label_dir)
y_train = yprep.y_preprocessing(y_train)
x_test  = dl.load_images(test_dir)
x_test0 = x_test
x_test = xprep.x_preprocessing(x_test)
img_shape = x_train[0].shape;
print(img_shape)
# img_shape = x_train.shape;
print(y_train.shape)

model = Sequential()
# Encoder Layers
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = img_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))           #(12,28)
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))                         #(6,14)
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())

# Decoder Layers
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2, 2)))                                         #(12,28)
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2, 2)))                                          #(24,56)
model.add(Dropout(0.25))
model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))

# model.add(Flatten())
# model.add(Dense(1344))
# model.add(Activation("relu"))
#
# # softmax classifier
# model.add(Dense(1344))
# model.add(Activation("softmax"))
# model.resize(img_shape)

model.summary()



batch_size=32
epochs= 10

"""
when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample). In order to convert integer targets into categorical targets, you can use the Keras utility to_categorical
"""
model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

# model.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['mse'])


model.fit(x_train, y_train,
           batch_size=batch_size, nb_epoch=epochs, verbose=1)

# H = model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
# 	validation_data=(y_train), steps_per_epoch=len(x_train) // BS,
# 	epochs=nb_epoch, verbose=1)

# model.save('model_(1400*600).h5')

y_pred = model.predict(x_test)
print(y_pred.shape)
# score = model.evaluate(x_test, y_pred, verbose=0)


plt.figure(1)
plt.imshow(x_test0[0])
plt.savefig('/Users/Hendrik/Desktop/x_test.png', bbox_inches='tight')
plt.show()


plt.figure(2)
plt.imshow(y_pred[0, :, :, 0])
# plt.savefig('/Users/Hendrik/Desktop/y_test.png', bbox_inches='tight')
plt.show()

plt.figure(2)
plt.imshow(y_pred[0, :, :, 1])
# plt.savefig('/Users/Hendrik/Desktop/y_test.png', bbox_inches='tight')
plt.show()


n= 6
# nx = 24
# ny = 56
nx = 192
ny = 448
# nx = 600
# ny = 1400
nz = 3
y = np.zeros((n, nx, ny, nz))
y[:,:,:,0:2] = y_pred
threshold, upper, lower = 2, 1, 0
y = np.where(y > threshold, upper, lower)
y= 255*y

plt.figure(4)
plt.imshow(y[0])
plt.show()

"""
https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/
# """

num_images = 6
# plt.figure(figsize=(56, 24))
plt.figure(figsize=(448, 192))
# plt.figure(figsize=(1400, 600))

for i in range(0,6):
   # plot original image
   ax = plt.subplot(2, num_images, i + 1)
   plt.imshow(x_test0[i])

   # plot reconstructed image
   ax = plt.subplot(2, num_images, num_images + i + 1)
   plt.imshow(y_pred[i, :, :, 0])

plt.savefig('/Users/Hendrik/Desktop/figurejh.png', bbox_inches='tight')
plt.show()

