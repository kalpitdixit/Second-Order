import os
import numpy as np

import theano
import theano.tensor as T
from theano.gof import Variable as V
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD
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
    output = Dense(cfg.output_dim,activation='softmax',name='outmax')(h2)
    model = Model(input=input_images,output=output)
    return model
    

def compile_model(model, cfg):
    sgd = SGD(lr=cfg.learning_rate, momentum=cfg.momentum)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return
    
def train(model, dataset, cfg):
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        all_labels = np.zeros((dataset.n_train, cfg.output_dim))
        all_labels[:,dataset.data['train_labels']] = 1
        history = model.fit(x = dataset.data['train_images'], 
                            y = all_labels, 
                            batch_size = cfg.batch_size,
                            epochs = 5)
        exit()
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            batch_labels = np.zeros((len(batch_inds), cfg.output_dim))
            batch_labels[:,dataset.data['train_labels'][batch_inds]] = 1
            history = model.fit(x = dataset.data['train_images'][batch_inds,:], 
                                y = batch_labels, 
                                batch_size = cfg.batch_size,
                                epochs = 1)
            #print type(history)
            #print history
            #exit()
                

if __name__=="__main__":
    data_dir = '/atlas/u/kalpit/data'
    dataset = Dataset(data_dir)
    cfg = Config()
    model = create_feedforward_classifier_model()
    model.summary()
    compile_model(model, cfg)
    train(model, dataset, cfg)
