import sys
import os
#save_dir = '/atlas/u/kalpit/Second-Order/code/mnist'
#sys.stdout = open(os.path.join(save_dir, 'stdout'), 'w')
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.gof import Variable as V
from theano.gof import Generic
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.backend import categorical_crossentropy
import theano.sandbox.cuda as cuda

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
        fname = os.path.join(save_dir, 'config.txt')
        with open(fname, 'w') as f:
            for k in self.__dict__:
                f.write(k+': '+str(self.__dict__[k])+'\n')


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
    elif cfg.optimizer=='adam':
        opt = 'adam'
    print 'Using Optimizer: {}'.format(cfg.optimizer)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return opt
    

def set_weights(model, computed_gradients, alpha):
    count = 0
    for l_num in range(len(model.layers)):
        wgts = model.layers[l_num].get_weights()
        n_wgts = len(wgts)
        wgts = [wgts[i]+alpha*computed_gradients[count+i] for i in range(n_wgts)]
        count += n_wgts
        model.layers[l_num].set_weights(wgts)
    return 


def train(model, opt, dataset, cfg):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss([], save_dir, 'max_learning_rates.txt', first_use=True)
    save_loss([], save_dir, 'learning_rates.txt', first_use=True)
    
    ##################
    gradients = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    input_tensors = [model.inputs[0], # input data
                     model.sample_weights[0], # how much to weight each sample by
                     model.targets[0], # labels
                     K.learning_phase(), # train or test mode
                    ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    ##################
    ##################
    #print [type(x) for x in model.trainable_weights]
    #exit()
    #alpha = 0
    #trainable_weights = model.trainable_weights
    ##updates = OrderedDict([(trainable_weights[i], trainable_weights[i]+alpha*gradients[i]) for i in range(len(trainable_weights))])
    #updates = [(trainable_weights[i], trainable_weights[i]+alpha*gradients[i]) for i in range(len(trainable_weights))]
    #print 'input check', np.any([isinstance(i, (list, tuple)) for i in [alpha]+ gradients])
    #update_weights = K.function(inputs=[], outputs=[], updates=updates)
    ##################

    print 'Beginning Iterations...'
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            batch_labels = np.zeros((len(batch_inds), cfg.output_dim)).astype(DTYPE)
            batch_labels[range(len(batch_inds)),dataset.data['train_labels'][batch_inds]] = 1
    
            ##################
            ## set alpha
            if epoch==0 and batch_num==0:
                alpha = 1e-2
                #a = theano._asarray(np.random.rand(*()), dtype='float32')
                #alpha = cuda.CudaNdarraySharedVariable(name='alpha',type=Generic(),strict=False,value=1e-2)
                #alpha.set_value(1e-2)
            ## get first gradient, g
            computed_gradients = get_gradients([dataset.data['train_images'][batch_inds,:], np.ones(len(batch_inds)), batch_labels, False])
            gT_g = np.sum([np.sum(np.square(g)) for g in computed_gradients])
            ## get f(x)
            fx, train_acc = model.evaluate(x = dataset.data['train_images'][batch_inds,:],
                                                     y = batch_labels,
                                                     batch_size = cfg.batch_size,
                                                     verbose = 0)   
            train_loss_batch.append(fx)
            train_acc_batch.append(train_acc)
            ## get f(x+alpaha*g)
            """
            update_weights([alpha, computed_gradients])
            #opt.lr.set_value(0.0)
            hist1 = model.fit(x = dataset.data['train_images'][batch_inds,:],
                             y = batch_labels,
                             batch_size = cfg.batch_size,
                             epochs = 1,
                             verbose = 0)   
            hist2 = model.fit(x = dataset.data['train_images'][batch_inds,:],
                             y = batch_labels,
                             batch_size = cfg.batch_size,
                             epochs = 1,
                             verbose = 0)   
            print hist1.history['loss'][0], hist2.history['loss'][0]
            exit()
            """
            set_weights(model, computed_gradients, alpha)
            fx_plus_ag, _ = model.evaluate(x = dataset.data['train_images'][batch_inds,:],
                                                     y = batch_labels,
                                                     batch_size = cfg.batch_size,
                                                     verbose = 0)

            ## get f(x-alpaha*g)
            set_weights(model, computed_gradients, -2*alpha)
            fx_minus_ag, _ = model.evaluate(x = dataset.data['train_images'][batch_inds,:],
                                                     y = batch_labels,
                                                     batch_size = cfg.batch_size,
                                                     verbose = 0)
            ## chosen learning rate
            gT_H_g = (fx_plus_ag + fx_minus_ag - 2*fx)/(alpha**2)
            max_lr = 2*gT_g / np.abs(gT_H_g)
            lr     = min(fx/gT_g, max_lr)
            save_loss([max_lr], save_dir, 'max_learning_rates.txt')
            save_loss([lr], save_dir, 'learning_rates.txt')
            
            ## print 
            print 'alpha             : ', alpha
            print 'f(x)              : ', fx
            print 'f(x)_plus_ag      : ', fx_plus_ag
            print 'f(x)_minus_ag     : ', fx_minus_ag
            print 'f(x+)+f(x-)-2f(x) : ', fx_plus_ag + fx_minus_ag - 2*fx
            print 'estimated (g.T)Hg : ', gT_H_g
            print 'gT_g              : ', gT_g
            print 'max_lr            : ', max_lr
            print 'chosen lr         : ', lr
            print 'epoch-batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            ## quit?
            if gT_H_g==0.0:
                break

            ## final update
            set_weights(model, computed_gradients, alpha-lr) # lr has to be -ve

            ## update alpha
            alpha = min(lr/2, 1e-1)
            
        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'epoch {} - average training cost: {:.3f}'.format(epoch+1, train_loss[-1])
        print ''
        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
        if math.isnan(train_loss[-1]):
            exit()
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
    final_run = False

    ## create unique run_id and related directory
    while True:
        run_id = np.random.randint(low=1000000000, high=9999999999) # 10 digits
        save_dir = os.path.join(os.getcwd(), 'output_'+str(run_id))
        if not os.path.exists(save_dir):
            break
    #run_id = run_id
    if not final_run:
        save_dir = '/atlas/u/kalpit/Second-Order/code/mnist/output'
    os.system('rm -rf '+save_dir)
    os.makedirs(save_dir)
   
    ## redirect stdout
    if final_run:
        sys.stdout = open(os.path.join(save_dir, 'stdout'), 'w')
    print run_id
    print 'testing'

    ## Data
    data_dir = '/scail/data/group/atlas/kalpit/data/mnist'
    dataset = Dataset(data_dir)

    ## Config
    cfg = Config(save_dir)
    
    ## Model
    print 'Creating Model...'
    model = create_feedforward_classifier_model()
    model.summary()
    opt = compile_model(model, cfg)

    ## Train
    print 'Training Model...'
    starttime = time.time()
    train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc = train(model, opt, dataset, cfg)
    endtime = time.time()
    plot_loss(train_loss, save_dir, 'training_cost', 'training_cost')
    plot_loss(val_loss, save_dir, 'validation_cost', 'validation_cost')

    ## Validate
    validate(model, dataset)
    
    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    print 'theano.config.device: ', theano.config.device
