import data_loader as dl
import os
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

main_dir = '/Users/Hendrik/OneDrive - ETHZ/Python/sensors_project/Sensors_project'
model_dir = '/Users/Hendrik/OneDrive - ETHZ/Python/biosensors_project/models_part/validation_split_0.15/segnetwork/1_epochs/Kernel=3/segnetwork3(1400*600).h5'
test_dir = os.path.join(main_dir, 'test_images')

"""
x_test:  original pictures without corresponding labeled pictures (used to create a visual evaluation of Network with 
         plotter)
"""
x_test = dl.load_images(test_dir, ny, nx)
x_test0 = x_test
x_test = dl.x_preprocessing(x_test)
print("x_test shape = "+str(x_test.shape))
xtest_chunks = dl.xtest_partitioning(x_test)   # 18 256*256 bits of x_test images


model = load_model(model_dir)
ypred_bits = model.predict(xtest_chunks)

y_pred= dl.ypred_reconstruct(ypred_bits)

print("y_pred shape= "+ str(y_pred.shape))
plt.figure(1)
plt.imshow(x_test0[0])
plt.show()

num_images = 6

nz = 3
y = np.zeros((num_images, ny, nx, nz))
y[:,:,:,0:2] = y_pred
plt.imshow(y[0])
plt.show()
threshold, upper, lower = 0.5, 1, 0
y = np.where(y > threshold, upper, lower)
y = 255 * y
print("y shape = " + str(y.shape))
for i in range(0, 6):
    plt.figure()
    plt.imshow(y[i])
    plt.show()

plt.figure(figsize=(ny, nx))

for i in range(0, 6):
    # plot original image
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(x_test0[i])

    # plot reconstructed image
    ax = plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(y[i])

plt.savefig('/Users/Hendrik/Desktop/th' + str(threshold) + '.png', bbox_inches='tight')
