import sys
#sys.stdout = open('/atlas/u/kalpit/Second-Order/code/mnist/output', 'w')

import os
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
print 'theano.config.device: ', theano.config.device
import theano.tensor as T
from theano.gof import Variable as V
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD, BB
from keras.backend import categorical_crossentropy

from data_handler import Dataset

class Config(object):
    def __init__(self):
        self.input_dim  = 784
        self.output_dim = 10
        self.max_epochs = 1
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.momentum = 0.99


def create_feedforward_classifier_model(cfg=Config()):
    input_images = Input(shape=(cfg.input_dim,), name='input_images')
    h1 = Dense(1000,activation='relu',name='h1')(input_images)
    h2 = Dense(1000,activation='relu',name='h2')(h1)
    output = Dense(cfg.output_dim,activation='softmax',name='softmax')(h2)
    model = Model(input=input_images,output=output)
    return model
    

def compile_model(model, cfg):
    #sgd = SGD(lr=cfg.learning_rate, momentum=cfg.momentum)
    bb  =  BB()
    model.compile(optimizer=bb,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return
    

def train(model, dataset, cfg):
    train_loss = [] # record_loss
    train_acc = [] # record_accuracy
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            batch_labels = np.zeros((len(batch_inds), cfg.output_dim))
            batch_labels[range(len(batch_inds)),dataset.data['train_labels'][batch_inds]] = 1
            history = model.fit(x = dataset.data['train_images'][batch_inds,:], 
                                y = batch_labels, 
                                batch_size = cfg.batch_size,
                                epochs = 1,
                                verbose = 0)
            train_loss.append(history.history['loss'][0])
            train_acc.append(history.history['acc'][0])
            print 'Epoch-Batch: {:2d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,train_loss[-1],train_acc[-1])  
    return train_loss, train_acc


def validate(model, dataset):
    val_labels = np.zeros((dataset.n_val,cfg.output_dim))
    val_labels[range(dataset.n_val),dataset.data['val_labels']] = 1
    val_loss, val_acc = model.evaluate(x = dataset.data['val_images'],
                                       y = val_labels,
                                       batch_size = cfg.batch_size,
                                       verbose = 0)
    print 'validation_loss: {:.3f}  validation_acc:{:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc
             

def plot_train_loss(train_loss):
    plt.figure()
    plt.semilogy(train_loss)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('training cost')
    plt.savefig('plot_training_cost')
    return


if __name__=="__main__":
    ## Data
    data_dir = '/atlas/u/kalpit/data'
    dataset = Dataset(data_dir)

    ## Config
    cfg = Config()

    ## Model
    model = create_feedforward_classifier_model()
    model.summary()
    compile_model(model, cfg)

    ## Train
    starttime = time.time()
    train_loss, train_acc = train(model, dataset, cfg)
    endtime = time.time()
    plot_train_loss(train_loss)
    validate(model, dataset)

    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    print 'theano.config.device: ', theano.config.device
