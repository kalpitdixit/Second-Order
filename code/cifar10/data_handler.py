import os
import numpy as np

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.get_data()
        self.n_train = self.data['train_images'].shape[0]
        self.n_val   = self.data['val_images'].shape[0]
        self.n_test  = self.data['test_images'].shape[0]

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
        return X, Y

    def load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)    
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
      return Xtr, Ytr, Xte, Yte

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

        Xtr, Ytr, Xte, Yte = self.load_CIFAR10(self.data_dir)
        data = {}

        ### training
        data['train_images'] = Xtr 
        data['train_labels'] = Ytr
        ### validation
        data['val_images'] = Xte
        data['val_labels'] = Yte
        #### testing
        #data['test_images'] = np.load(os.path.join(self.data_dir, 'testing', 'images.npy'))
        #data['test_labels'] = np.load(os.path.join(self.data_dir, 'testing', 'labels.npy'))
        return data

