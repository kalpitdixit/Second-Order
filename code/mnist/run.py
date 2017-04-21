import os
import numpy as np

import theano
import theano.tensor as T
import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model
import lasagne

from utils import *

class Config():
    def __init__(self):
        self.input_dim = 784

def create_feedforward_classifier_model(cfg=Config()):
    input_images = Input(shape=(None,cfg.input_dim), name='input_images')
    h1 = Dense(1000,activation='relu',name='h1')(input_images)
    h2 = Dense(1000,activation='relu',name='h2')(h1)
    pre_softmax = Dense(10,activation='linear',name='pre_softmax')(h2)
    output = Activation('softmax')(pre_softmax)
    model = Model(input=input_images, output=output)
    return model
    

def build_train_fn(model):
    ### cost
    lr = T.scalar()
    input_images = model.inputs[0]
    labels = K.placeholder(ndim=2, dtype='int32')

    softmax_outputs = model.outputs[0]
    cost = categorical_crossentropy(softmax_outputs, labels).mean()

    ### gradients
    trainable_vars = model.trainable_weights
    grads = K.gradients(cost, trainable_vars)
    grads = lasagne.updates.total_norm_constraint(grads, 100)
    updates = lasagne.updates.nesterov_momentum(grads, trainable_vars, lr, momentum=0.99)
    for key, val in model.updates: # needed like this to update for batch normalization
        updates[key] = val

    ### train_fn
    train_fn = K.function([input_images, labels, K.learning_phase(), lr],
                          [softmax_outputs, cost],
                          updates=updates)

    return train_fn
    
    

if __name__=="__main__":
    data_dir = '/atlas/u/kalpit/data'
    #data = get_data(data_dir)
    model = create_feedforward_classifier_model()
    model.summary()
    train_fn = build_train_fn(model)
