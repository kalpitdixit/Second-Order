import sys
import os
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.gof import Variable as V
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD, BB, BB_momentum, Adam, Adadelta_momentum
from keras.backend import categorical_crossentropy

from data_handler import Dataset

DTYPE = 'float32'

class Config(object):
    def __init__(self, save_dir=None):
        self.input_dim  = 784
        self.output_dim = 10
        self.max_epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-1
        self.momentum = 0.0
        self.optimizer = 'sgd'
        self.base_lr = 1.0
        self.per_param = True
        self.use_abs = False
        self.lbound = 0 # -1e100
        self.ubound = 1e0 #1e100
        self.rho = 0.95
        if save_dir is not None:
            self.save(save_dir)

    def save(self, save_dir):
        fname = os.path.join(save_dir, 'config.pkl') 
        with open(fname, 'w') as f:
            pickle.dump(self.__dict__, f, 2)


def create_feedforward_classifier_model(cfg=Config()):
    input_images = Input(shape=(cfg.input_dim,), name='input_images')
    h1 = Dense(1000,activation='relu',name='h1')(input_images)
    h1 = Dropout(0.5,name='d1')(h1) ####
    h2 = Dense(1000,activation='relu',name='h2')(h1)
    h2 = Dropout(0.5,name='d2')(h2) ####
    #output = Dense(cfg.output_dim,activation='softmax',name='softmax')(h2)
    pre_final = Dense(cfg.output_dim,activation='linear',name='pre_final')(h2)
    output = Activation('softmax',name='softmax')(pre_final)
    model = Model(input=input_images,output=output)
    return model
    

def compile_model(model, cfg):
    if cfg.optimizer=='sgd':
        opt = SGD(lr=cfg.learning_rate, momentum=cfg.momentum)
    elif cfg.optimizer=='bb':
        opt = BB(base_lr=cfg.base_lr, per_param=cfg.per_param, use_abs=cfg.use_abs, lbound=cfg.lbound, ubound=cfg.ubound)
    elif cfg.optimizer=='adam':
        opt = 'adam'
    elif cfg.optimizer=='aa':
        opt = 'aa'
    elif cfg.optimizer=='bb_mom':
        opt = BB_momentum(base_lr=cfg.base_lr, per_param=cfg.per_param, use_abs=cfg.use_abs, 
                          lbound=cfg.lbound, ubound=cfg.ubound, rho=cfg.rho)
    elif cfg.optimizer=='adadelta':
        opt = 'adadelta'
    elif cfg.optimizer=='adadelta_mom':
        opt = Adadelta_momentum(momentum=cfg.momentum)
    print 'Using Optimizer: {}'.format(cfg.optimizer)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return
    

def train(model, dataset, cfg):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            batch_labels = np.zeros((len(batch_inds), cfg.output_dim)).astype(DTYPE)
            batch_labels[range(len(batch_inds)),dataset.data['train_labels'][batch_inds]] = 1
            history = model.fit(x = dataset.data['train_images'][batch_inds,:], 
                                y = batch_labels, 
                                batch_size = cfg.batch_size,
                                epochs = 1,
                                verbose = 0)
            train_loss_batch.append(history.history['loss'][0])
            train_acc_batch.append(history.history['acc'][0])
            #print '='*100
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            #lrs = model.optimizer.get_weights()[2*6:3*6]
            #gcurr = model.optimizer.get_weights()[3*6:4*6]

            #shapes = [x.shape for x in model.optimizer.get_weights()[:6]]
            #print 'lrs          :', [np.sum(np.abs(x)) for x in lrs]
            #print 'len(model.optimizer.get_weights()): ', len(model.optimizer.get_weights())
            #print 'model.weights: ', [np.sum(np.abs(x.eval())) for x in model.weights]
            #print 'gcurr        : ', [np.sum(np.abs(x)) for x in gcurr]
            # with a Sequential model
            #print np.sum(layer_n_output, axis=0)
            #print np.shape(np.sum(layer_n_output, axis=0))
            #print np.count_nonzero(np.sum(layer_n_output, axis=0))
            #print sorted(model.trainable_weights, key=lambda x: x.name if x.name else x.auto_name)
            #print ''
            #print ''
        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc


def validate(model, dataset):
    val_labels = np.zeros((dataset.n_val,cfg.output_dim))
    val_labels[range(dataset.n_val),dataset.data['val_labels']] = 1
    val_loss, val_acc = model.evaluate(x = dataset.data['val_images'],
                                       y = val_labels,
                                       batch_size = cfg.batch_size,
                                       verbose = 0)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc
             

def save_loss(losses, save_dir, fname, first_use=False):
    if first_use:
        f = open(os.path.join(save_dir, fname), 'w')
    else:
        f = open(os.path.join(save_dir, fname), 'a')
    for loss in losses:
        f.write(str(loss)+'\n')
    f.close()
    return


def plot_loss(losses, save_dir, plotname, title=''):
    plt.figure()
    plt.semilogy(train_loss)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('training cost')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, 'plot_'+plotname))
    return


if __name__=="__main__":
    ## gpu_run?
    final_run = True

    ## create unique run_id and related directory
    while True:
        run_id = np.random.randint(low=1000000000, high=9999999999) # 10 digits
        save_dir = '/atlas/u/kalpit/Second-Order/code/mnist/output_' + str(run_id)
        if not os.path.exists(save_dir):
            break
    #run_id = run_id
    #save_dir = '/atlas/u/kalpit/Second-Order/code/mnist/output'
    os.system('rm -rf '+save_dir)
    os.makedirs(save_dir)
   
    ## redirect stdout
    if final_run:
        sys.stdout = open(os.path.join(save_dir, 'stdout'), 'w')
    print run_id
    print 'testing'

    ## Data
    data_dir = '/atlas/u/kalpit/data'
    dataset = Dataset(data_dir)

    ## Config
    cfg = Config(save_dir)
    
    ## Model
    model = create_feedforward_classifier_model()
    model.summary()
    compile_model(model, cfg)

    ## Train
    starttime = time.time()
    train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc = train(model, dataset, cfg)
    endtime = time.time()
    plot_loss(train_loss, save_dir, 'training_cost', 'training_cost')
    plot_loss(val_loss, save_dir, 'validation_cost', 'validation_cost')

    ## Validate
    validate(model, dataset)
    
    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    print 'theano.config.device: ', theano.config.device
