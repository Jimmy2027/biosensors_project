
#!/usr/bin/env python3
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


def load_images(data_dir, nx, ny):
    im_data = []
    images = [o for o in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir,o)) and not
                    o.startswith('.')]


    images.sort()                       #does this make sense????
    # np.sort(images)



    for image in images:
        image_path = os.path.join(data_dir, image)
        im = imageio.imread(image_path)
        im = im[:,:,0:3]
        res = cv2.resize(im, dsize=(ny, nx))
        im_data.append(res)


    return np.array(im_data)




""" I am training the network on the green and red part of the RGB image. I guess that the blue part contains no information that is needed for the network
"""

def x_preprocessing(x):

    x = x[:, :, :,  0:2]

    x = x / 255.

    return x




def y_preprocessing(y):

    y = y[:, :, :,  0:2]
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
    #mask_datagen.fit(label_array, augment=True, seed=seed)
    #mask_datagen_g.fit(label_array_g, augment=True, seed=seed)
    print('Augmentation worked')
    return image_datagen, image_array, label_array




