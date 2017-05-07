import os
import numpy as np
import pickle

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.get_data()
        self.n_train = self.data['train_images'].shape[0]
        self.n_val   = self.data['val_images'].shape[0]
        print 'USING "TEST" DATA AS VAL!!!'
        #self.n_test  = self.data['test_images'].shape[0]

    def get_data(self):
        """
        "data" - dict
        data['train_images'] - 2d np array. data['train_images'][i] refers to train image number i. 2nd dim is 784.
        data['train_labels'] - 1d list. data['train_labels'][i] refers to label of train image number i.
        data['val_images'] - 2d np array. data['val_images'][i] refers to val image number i. 2nd dim is 784.
        data['val_labels'] - 1d list. data['val_labels'][i] refers to label of val image number i.
        data['test_images'] - 2d np array. data['test_images'][i] refers to test image number i. 2nd dim is 784.
        data['test_labels'] - 1d list. data['test_labels'][i] refers to label of test image number i.
        """

        data = {}

        ### training
        data['train_images'] = np.load(os.path.join(self.data_dir, 'training', 'images.npy'))
        data['train_labels'] = np.load(os.path.join(self.data_dir, 'training', 'labels.npy'))
        ### validation
        data['val_images'] = np.load(os.path.join(self.data_dir, 'validation', 'images.npy'))
        data['val_labels'] = np.load(os.path.join(self.data_dir, 'validation', 'labels.npy'))
        #### testing
        #data['test_images'] = np.load(os.path.join(self.data_dir, 'testing', 'images.npy'))
        #data['test_labels'] = np.load(os.path.join(self.data_dir, 'testing', 'labels.npy'))

        return data

