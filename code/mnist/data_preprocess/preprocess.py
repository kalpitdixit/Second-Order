import os
import shutil
import sys
sys.path.append("/atlas/u/kalpit/mnist")

from mnist import MNIST
import numpy as np

"""
Steps of preprocessing done:
1. read all image files from the "mnist" directory. training and testing.
2. save each file as a separate .npy file
3. compute the mean image
4. read all npy files, subtract the mean training image from them and write them back
"""
DTYPE='float32'

def preprocess_data(data, dir_images, dir_labels, mean_image=None):
    calc_mean_image = True # use the mean_image from the training set
    if mean_image is not None:
        calc_mean_image = False
    else:
        mean_image = np.zeros((28,28)).astype(DTYPE)
        num_images = 0
    ## use image files from mnist and store as npy. Compute and store mean image.
    for i in range(len(data[0])):
        if i%10000==0:
            print i
        image = np.asarray(data[0][i]).astype(DTYPE)
        image = image.reshape(28,28)
        np.save(os.path.join(dir_images, 'image_{}.npy'.format(i)), image)
        if calc_mean_image:
            num_images += 1
            mean_image += image
    if calc_mean_image:
        mean_image /= num_images
        np.save(os.path.join(dir_images, 'image_mean.npy'), mean_image)
    ## read all .npy image files and subtract mean_image from them. save each .npy file
    for i in range(len(data[0])):
        if i%10000==0:
            print i
        image = np.load(os.path.join(dir_images, 'image_{}.npy'.format(i)))
        image -= mean_image
        np.save(os.path.join(dir_images, 'image_{}.npy'.format(i)), image)
    ## use labels from mnist and store as single .txt file. 
    f = open(os.path.join(dir_labels, 'labels.txt'), 'w')
    for i in range(len(data[0])):
        f.write(str(data[1][i])+'\n')
    f.close()
    return mean_image

## important directories
dir_raw_data = '/atlas/u/kalpit/mnist/data'
dir_train_images = '/atlas/u/kalpit/data/training/images'
dir_train_labels = '/atlas/u/kalpit/data/training/labels'
dir_test_images = '/atlas/u/kalpit/data/testing/images'
dir_test_labels = '/atlas/u/kalpit/data/testing/labels'
for d in [dir_train_images, dir_train_labels, dir_test_images, dir_test_labels]:
    os.system('rm -rf '+d)
    os.makedirs(d)

### read TRAINING images from "python-mnist" and save as numpy files
mndata = MNIST(dir_raw_data)
data_training = mndata.load_training() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.
data_testing = mndata.load_testing() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.

mean_image = preprocess_data(data_training, dir_train_images, dir_train_labels)
preprocess_data(data_testing, dir_test_images, dir_test_labels, mean_image=mean_image)
