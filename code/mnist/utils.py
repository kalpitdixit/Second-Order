import os
import numpy as np

def get_images_and_labels(data_dir):
    images = np.load(os.path.join(data_dir, 'images.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    return images, labels

def get_data(data_dir):
    """
    returns a dict called "data"
    data['train_images'] - 2d np array. data['train_images'][i] refers to train image number i. 2nd dim is 784.
    data['train_labels'] - 1d list. data['train_labels'][i] refers to label of train image number i.
    data['test_images'] - 2d np array. data['test_images'][i] refers to test image number i. 2nd dim is 784.
    data['test_labels'] - 1d list. data['test_labels'][i] refers to label of test image number i.
    """

    data = {}
    ### training
    data['train_images'] = np.load(os.path.join(data_dir, 'training', 'images.npy'))
    data['train_labels'] = np.load(os.path.join(data_dir, 'training', 'labels.npy'))
    ### testing
    data['test_images'] = np.load(os.path.join(data_dir, 'testing', 'images.npy'))
    data['test_labels'] = np.load(os.path.join(data_dir, 'testing', 'labels.npy'))
    return data
