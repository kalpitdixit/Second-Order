import os
import shutil
import sys
sys.path.append("/atlas/u/kalpit/mnist")

from mnist import MNIST
import numpy as np

"""
Steps of preprocessing done:
1. read all image files from the "mnist" directory. training and testing.
2. divide all pixels by 255 to get each pixel in the range [0,1]
3. images = images - mean_image. mean_image is passed (Test Set) or computed (Train Set)
4. save a single file of all images as .npy
5. save a single file of all labels as .npy
"""
DTYPE='float32'

def preprocess_data(data, store_dir, mean_image=None):
    num_images = len(data[0])
    data_images = np.zeros((num_images, 784))
    ## use image files from mnist and store as npy. Compute and store mean image.
    for i in range(len(data[0])):
        data_images[i,:] = np.asarray(data[0][i]).astype(DTYPE)/255
    if mean_image is None:
        mean_image = np.mean(data_images, axis=0)
        np.save(os.path.join(store_dir, 'image_mean.npy'), mean_image)
    data_images = data_images - mean_image
    np.save(os.path.join(store_dir, 'images.npy'), data_images)
    ## use labels from mnist and store as single .npy file. 
    data_labels = np.zeros(num_images)
    for i in range(len(data[0])):
        data_labels[i] = int(data[1][i])
    np.save(os.path.join(store_dir, 'labels.npy'), data_labels)
    ## create "info" file
    with open(os.path.join(store_dir, 'info'), 'w') as f:
        f.write(str(num_images))
    return mean_image

## important directories
dir_raw_data = '/atlas/u/kalpit/mnist/data'
dir_train = '/atlas/u/kalpit/data/training'
dir_test  = '/atlas/u/kalpit/data/testing'
for d in [dir_train, dir_test]:
    os.system('rm -rf '+d)
    os.makedirs(d)

### read TRAINING images from "python-mnist" and save as numpy files
mndata = MNIST(dir_raw_data)
data_training = mndata.load_training() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.
data_testing = mndata.load_testing() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.

mean_image = preprocess_data(data_training, dir_train)
preprocess_data(data_testing, dir_test)
