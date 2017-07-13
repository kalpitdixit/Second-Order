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
    data_images = np.zeros((num_images, 784)).astype(DTYPE)

    ## use image files from mnist and store as npy. Compute and store mean image.
    for i in range(len(data[0])):
        data_images[i,:] = np.asarray(data[0][i]).astype(DTYPE)/255
    if mean_image is None:
        mean_image = np.mean(data_images, axis=0).astype(DTYPE)
        np.save(os.path.join(store_dir, 'image_mean.npy'), mean_image)
    data_images = data_images - mean_image
    np.save(os.path.join(store_dir, 'images.npy'), data_images)

    ## use labels from mnist and store as single .npy file. 
    data_labels = np.zeros(num_images).astype('int32')
    for i in range(len(data[0])):
        data_labels[i] = int(data[1][i])
    np.save(os.path.join(store_dir, 'labels.npy'), data_labels)

    ## create "info" file
    with open(os.path.join(store_dir, 'info'), 'w') as f:
        f.write(str(num_images))

    return mean_image

if __name__=='__main__':
    ### important directories
    dir_raw_data = '/atlas/u/kalpit/mnist/data'
    dir_train = '/atlas/u/kalpit/data/training'
    dir_val   = '/atlas/u/kalpit/data/validation'
    dir_test  = '/atlas/u/kalpit/data/testing'
    for d in [dir_train, dir_val, dir_test]:
        os.system('rm -rf '+d)
        os.makedirs(d)

    ### number of validatio images
    num_val_images = 10000 # total data is train: 60,000 and test: 10,000

    ### read TRAINING and TESTING images from "python-mnist"
    mndata = MNIST(dir_raw_data)
    data_training = mndata.load_training() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.
    data_testing = mndata.load_testing() # dt is a 2D list of length 2. dt[0] is a list of 60,000 images. dt[1] is a list of 60,000 labels.
    n_train = len(data_training[0])

    ### split TRAINING images into TRAINING and VALIDATION sets
    data_validation = [data_training[0][n_train-num_val_images:], data_training[1][n_train-num_val_images:]]
    data_training   = [data_training[0][:n_train-num_val_images], data_training[1][:n_train-num_val_images]]

    ### save TRAINING, VALIDATION and TESTING images as numpy files
    mean_image = preprocess_data(data_training, dir_train) # use mean_image from train set
    preprocess_data(data_validation, dir_val, mean_image=mean_image)
    preprocess_data(data_testing, dir_test, mean_image=mean_image)
