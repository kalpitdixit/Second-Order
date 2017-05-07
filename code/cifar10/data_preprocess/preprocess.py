import os
import numpy as np
import pickle

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xval, Yval = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xval, Yval

def mean_subtraction_and_scaling(Xtr, Xval):
    ## scaling
    Xtr  /= 255 
    Xval /= 255 

    ## mean image
    mean_image = np.mean(Xtr, axis=0)

    ## mean subtraction
    Xtr  -= mean_image
    Xval -= mean_image
    return Xtr, Xval

if __name__=='__main__':
    dir_raw_data = '/scail/data/group/atlas/kalpit/data/cifar10'
    dir_train    = '/scail/data/group/atlas/kalpit/data/cifar10/training'
    dir_val      = '/scail/data/group/atlas/kalpit/data/cifar10/validation'
    for d in [dir_train, dir_val]:
        os.system('rm -rf '+d)
        os.makedirs(d)
    print 'USING "TEST" DATA AS VAL!!!'

    Xtr, Ytr, Xval, Yval = load_CIFAR10(dir_raw_data)
    Xtr, Xval = mean_subtraction_and_scaling(Xtr, Xval)

    ## save data
    np.save(os.path.join(dir_train, 'images.npy'), Xtr)
    np.save(os.path.join(dir_train, 'labels.npy'), Ytr)
    np.save(os.path.join(dir_val, 'images.npy'), Xval)
    np.save(os.path.join(dir_val, 'labels.npy'), Yval)
        

