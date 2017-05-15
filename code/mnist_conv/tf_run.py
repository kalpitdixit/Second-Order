import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
import tensorflow.contrib.layers as layers

from data_handler import Dataset
from common.models import get_model
from common.utils import save_loss, plot_loss

DTYPE = 'float32'

class Config(object):
    def __init__(self):
        self.input_height = 28
        self.input_width  = 28
        self.input_nchannels = 1
        self.output_dim = 10
        self.keep_prob  = 0.8

        self.max_epochs = 20
        self.batch_size = 128
        self.learning_rate = 1e-1 # 1e-1 for sgd, 1e-3 for adam
        self.beta1 = 0.99  # for adam
        self.beta2 = 0.999 # for adam
        self.rho = 0.1 # for adadelta
        self.momentum = 0.9
        self.nesterov = True
        self.early_stopping = 10
        self.optimizer = 'adadelta'
        
    def save(self, save_dir):
        fname = os.path.join(save_dir, 'config.pkl') 
        with open(fname, 'w') as f:
            pickle.dump(self.__dict__, f, 2)
        fname = os.path.join(save_dir, 'config.txt') 
        with open(fname, 'w') as f:
            for k in self.__dict__:
                f.write(k+': '+str(self.__dict__[k])+'\n')


def train(model, dataset, cfg):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    time_since_improvement = 0
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            feed_dict = {model.input_images: dataset.data['train_images'][batch_inds,:],
                         model.labels: dataset.data['train_labels'][batch_inds],
                         model.lr: cfg.learning_rate,
                         model.use_past_bt: False,
                         model.input_past_bt: np.zeros((len(batch_inds),model.cfg.input_height,model.cfg.input_width,model.cfg.input_nchannels)),
                         model.fc4_past_bt: np.zeros((len(batch_inds),1000))
                        }
            loss, acc, _ = model.sess.run([model.loss, model.accuracy, model.train_op], feed_dict=feed_dict)
            train_loss_batch.append(loss)
            train_acc_batch.append(acc)
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        if train_loss[-1] == min(train_loss):
            time_since_improvement  = 0
        else:
            time_since_improvement += 1
            if time_since_improvement >= cfg.early_stopping:
                print 'early stopping. no improvement since ', str(cfg.early_stopping), ' epochs.'
                break
                
        vl, va = validate(model, dataset)
        val_loss.append(vl)
        val_acc.append(va)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc


def validate(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
                 model.use_past_bt: False,
                 model.input_past_bt: np.zeros((dataset.n_val,model.cfg.input_height,model.cfg.input_width,model.cfg.input_nchannels)),
                 model.fc4_past_bt: np.zeros((dataset.n_val,1000))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc


def get_save_dir(cfg):
    save_dir = os.path.join(os.getcwd(), 'results', cfg.optimizer)
    nfiles_fname = os.path.join(save_dir,'num_files')
    if os.path.exists(nfiles_fname):
        with open(nfiles_fname, 'r') as f:
            nfiles = int(f.readline().strip())
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        nfiles = 0
    nfiles += 1
    while os.path.exists(os.path.join(save_dir, 'run_'+str(nfiles))):
        nfiles += 1
    with open(nfiles_fname, 'w') as f:
        f.write(str(nfiles))
    save_dir = os.path.join(save_dir, 'run_'+str(nfiles))
    os.makedirs(save_dir)
    print 'save_dir: ', save_dir
    return save_dir


def main(dataset_name, save_dir, cfg):
    ## Data
    data_dir = os.path.join('/scail/data/group/atlas/kalpit/data', dataset_name)
    dataset = Dataset(data_dir)
    dataset.data_reshape((cfg.input_height,cfg.input_width,cfg.input_nchannels))

    ## Model
    print 'Creating Model...'
    model = get_model(dataset_name+'_conv', cfg)
    #model.summary()

    ## Train
    print 'Training Model...'
    starttime = time.time()
    train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc = train(model, dataset, cfg)
    endtime = time.time()
    #plot_loss(train_loss, save_dir, 'training_cost', 'training_cost')
    #plot_loss(val_loss, save_dir, 'validation_cost', 'validation_cost')

    ## Validate
    print ''
    print 'Final Validation...'
    validate(model, dataset)
    
    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    return min(train_loss)

if __name__=="__main__":
    ## dataset
    dataset_name = 'mnist'

    ## gpu_run?
    final_run = True

    # adadelta
    #lr = [1e-4 ,1e-3, 1e-2, 1e-1]
    #rho = [0.8, 0.9, 0.95, 0.99]
    #params = list(itertools.product(lr, rho))
    params = [(0.1, 0.99)]

    # sgd nesterov
    #lr = [1e-4 ,1e-3, 1e-2, 1e-1]
    #mom = [0.9, 0.95, 0.99]
    #params = list(itertools.product(lr, mom))
    #params = [(1e-2,0.99)]

    # adam
    #lr = [1e-4, 1e-3, 1e-2]
    #beta1 = [0.9, 0.95, 0.99, 0.995]
    #beta2 = [0.9, 0.99, 0.999, 0.9999]
    #params = list(itertools.product(lr, beta1, beta2))
    #params = [(0.0001, 0.9, 0.99)] + params[19:]

    best_loss = float('inf')
    best_run_num = None
    best_params = None

    for i in range(len(params)):
        tf.reset_default_graph()
        print 'now running params: ', params[i]

        ## Config
        cfg = Config()
        # sgd
        #cfg.optimizer = 'sgd'
        #cfg.max_epochs = 200
        #cfg.nesterov = True
        #cfg.learning_rate = params[i][0] # 1e-1 for sgd
        #cfg.mom = params[i][1] # for sgd
        #cfg.keep_prob = 0.8
        
        # adadelta
        cfg.optimizer = 'adadelta'
        cfg.learning_rate = params[i][0] 
        cfg.rho = params[i][1]
        cfg.keep_prob = 0.8
        cfg.max_epochs = 200
    
        # adam
        #cfg.optimizer = 'adam'
        #cfg.max_epochs = 20
        #cfg.learning_rate = params[i][0] 
        #cfg.beta1 = params[i][1] 
        #cfg.beta2 = params[i][2]

        ## get save_dir: (redirect stdout, save config) to save_dir
        save_dir = get_save_dir(cfg)
        if final_run:
            sys.stdout = open(os.path.join(save_dir, 'stdout'), 'w')
        cfg.save(save_dir)

        ## main
        min_loss = main(dataset_name, save_dir, cfg)

        ## print to console
        sys.stdout = sys.__stdout__
        print params[i], ' : min_loss : ', min_loss
        
        ## best loss?
        if best_loss > min_loss:
            best_loss = min_loss
            best_save_dir = save_dir
            best_params = params[i]

        print ''
        print 'best_loss     : ', best_loss
        print 'best_save_dir : ', best_save_dir
        print 'best_params   : ', best_params
        print ''
