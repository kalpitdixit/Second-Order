import os
import numpy as np

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.get_data()
        self.n_train = self.data['train_images'].shape[0]
        self.n_test  = self.data['test_images'].shape[0]

    def get_data(self):
        """
        "data" - dict
        data['train_images'] - 2d np array. data['train_images'][i] refers to train image number i. 2nd dim is 784.
        data['train_labels'] - 1d list. data['train_labels'][i] refers to label of train image number i.
        data['test_images'] - 2d np array. data['test_images'][i] refers to test image number i. 2nd dim is 784.
        data['test_labels'] - 1d list. data['test_labels'][i] refers to label of test image number i.
        """

        data = {}
        ### training
        data['train_images'] = np.load(os.path.join(self.data_dir, 'training', 'images.npy'))
        data['train_labels'] = np.load(os.path.join(self.data_dir, 'training', 'labels.npy'))
        ### testing
        data['test_images'] = np.load(os.path.join(self.data_dir, 'testing', 'images.npy'))
        data['test_labels'] = np.load(os.path.join(self.data_dir, 'testing', 'labels.npy'))
        return data

