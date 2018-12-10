# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:57:18 2018

@author: Hendrik
"""
import numpy as np
import imageio
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from keras.utils import np_utils
import os
import random


def load_images(data_dir, nx, ny):
    im_data = []
    images = [o for o in os.listdir(data_dir)
              if os.path.isfile(os.path.join(data_dir, o)) and not
              o.startswith('.')]

    images.sort()
    # np.sort(images)

    for image in images:
        image_path = os.path.join(data_dir, image)
        im = imageio.imread(image_path)
        im = im[:, :, 0:3]
        res = cv2.resize(im, dsize=(ny, nx))
        im_data.append(res)

    return np.array(im_data)


""" I am training the network on the green and red part of the RGB image. I guess that the blue part contains no information that is needed for the network
"""


def x_preprocessing(x):
    x = x[:, :, :, 0:2]

    x = x / 255.

    return x


def y_preprocessing(y):
    y = y[:, :, :, 0:2]
    y = np.around(y / 255.)

    return y


def training_data_generator(image_array_r, image_array_g, label_array_r, label_array_g):  # , images, labels
    image_array = np.append(image_array_r, image_array_g, axis=0)
    label_array = np.append(label_array_r, label_array_g, axis=0)
    data_gen_args = dict(rotation_range=0,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.2,
                         vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    print(image_array.dtype)
    image_datagen.fit(image_array, augment=True, seed=seed)
    # image_datagen_r.fit(image_array_g, augment=True, seed=seed)
    # mask_datagen.fit(label_array, augment=True, seed=seed)
    # mask_datagen_g.fit(label_array_g, augment=True, seed=seed)
    print('Augmentation worked')
    return image_datagen, image_array, label_array


"""
nx: 128 bis (600-128)=472 
ny: 128 bis (1400-128)=1272
-> take random voxel and get 256*256 entourage
"""


def create_random_imagepart(x_train, y_train):
    t_range = 50
    xtrain_chunk = np.empty((33, 256, 256, 2))
    ytrain_chunk = np.empty((33, 256, 256, 2))
    x_training_data = np.empty((33 * t_range, 256, 256, 2))
    y_training_data = np.empty((33 * t_range, 256, 256, 2))

    for i in range(0, 33):
        for t in range(1, t_range):
            nx = random.randint(128, 472)
            ny = random.randint(128, 1272)
            xtrain_chunk[i, :, :, :] = x_train[i, nx - 128:nx + 128, ny - 128:ny + 128, :]
            ytrain_chunk[i, :, :, :] = y_train[i, nx - 128:nx + 128, ny - 128:ny + 128, :]
            np.append(x_training_data, xtrain_chunk, axis=0)
            np.append(y_training_data, ytrain_chunk, axis=0)
    return [x_training_data, y_training_data]


# TODO create zones so that at least one 256*256 part is taken from each zone (to have a less bigger likelihood that parts of the image are not taken)


"""
=> need to partition x_test into (3*6 = 18)*6 (256*256) parts with overlap
y - direction: 3 * 256*256 chunks (last one 344-600)
x - direction: 6 * 256*256 chunks (last one 1144-1400)
y_middles = 128, 128+256=384, 600-128= 472
x_middles = 128, 384, 640, 896, 1152, 1400-128=1272
"""


def xtest_partitioning(x_test):
    x_middles = [128, 384, 640, 896, 1152, 1272]
    y_middles = [128, 384, 472]                                 #middles of each of the 108 chunks
    xtest_chunks = np.empty((108, 256, 256, 2))
    i=0
    for t in range(0,5):

        for x in x_middles:
                for y in y_middles:
                    xtest_chunks[i, :, :, :] = x_test[t, x - 128:x + 128, y - 128:y + 128, :]
                    i=i+1
    print(xtest_chunks.shape)
    return xtest_chunks


"""
ypred_bits are stored such that: for every x_middle, 3 different y_middle values
y_middles = 128, 128+256=384, 600-128= 472
x_middles = 128, 384, 640, 896, 1152, 1400-128=1272
"""
def ypred_reconstruct(ypred_bits):

    ypred = np.empty((6, 600, 1400, 2))
    t = 0
    for i in range(0, 5):
        for x in range(0, 5):
            for y in range(0, 2):               # ypred_bits has shape (108, 256, 256, 2) with 108 = 6*3*6
                if y == 2:
                    ypred[i, x * 128:x * 128 + 256, 344:600, :] = ypred_bits[t, :, :, :]                    # have to take into account overlap
                if x == 5:
                    ypred[i, 1144:1400, y * 128:y * 128 + 256, :] = ypred_bits[t, :, :, :]
                ypred[i, x * 128:x * 128 + 256, y * 128:y * 128 + 256, :] = ypred_bits[t, :, :, :]   #ERROR could not broadcast input array from shape (256,256,2) into shape (216,256,2)
                t = t+1
                print(t)

    return ypred
