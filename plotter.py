import data_loader as dl
import os
import image_plotter
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import plot_model

"""
resolution of images:
"""
# nx = 24
# ny = 56
# nx = 192
# ny = 448
nx = 1400
ny = 600
nz = 3          #RGB values

main_dir = '/Users/Hendrik/OneDrive - ETHZ/Python/sensors_project/Sensors_project'
model_dir = '/Users/Hendrik/OneDrive - ETHZ/Python/biosensors_project/models_part/validation_split_0.25/segnetwork/100_epochs/Kernel=3/segnetwork3(1400*600).h5'
test_dir = os.path.join(main_dir, 'test_images')

"""
x_test:  original pictures without corresponding labeled pictures (used to create a visual evaluation of Network with 
         plotter)
"""
x_test = dl.load_images(test_dir, ny, nx)
x_test0 = x_test
x_test = dl.x_preprocessing(x_test)
xtest_chunks = dl.xtest_deconstruct(x_test)   # 18 256*256 bits of x_test images



model = load_model(model_dir)
ypred_bits = model.predict(xtest_chunks)
print("std of ypred_bits = " + str(np.std(ypred_bits)))
y_pred = dl.ypred_reconstruct(ypred_bits)

plt.figure(1)
plt.imshow(x_test0[0])
plt.show()

num_images = 6

nz = 3
y = np.zeros((num_images, ny, nx, nz))
y[:,:,:,0:2] = y_pred
test = y
threshold, upper, lower = 0.3, 1, 0
y = np.where(y > threshold, upper, lower)
y = 255 * y
print("y shape = " + str(y.shape))

for i in range(0, 6):
    plt.figure()
    plt.imshow(y[i])
    plt.show()

results = np.append(x_test0, y, axis=0)
image_plotter.show_images(results, threshold)
