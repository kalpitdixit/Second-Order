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
import argparse

import tensorflow as tf
import tensorflow.contrib.layers as layers

from data_handler import Dataset
from common.models_inbuilt import get_model
from train_funcs import train_ff_vanilla,          train_ff_kalpit          # train functions
from train_funcs import train_conv_vanilla,        train_conv_kalpit        # train functions
from train_funcs import train_autoencoder_vanilla, train_autoencoder_kalpit # train functions
from train_funcs import validate_ff, validate_conv, validate_autoencoder    # validation functions
from common.utils import save_loss, plot_loss, get_save_dir

DTYPE = 'float32'

class Config(object):
    def __init__(self, args):
        ## network definition
        if args.dataset=='mnist':
            if args.network=='ff':
                self.input_dim = 784
            elif args.network=='conv':
                self.input_height   = 28
                self.input_width    = 28
                self.input_nchannels = 1
            elif args.network=='autoencoder':
                self.input_dim = 784
            self.output_dim = 10
        if args.dataset=='cifar10':
            if args.network=='conv':
                self.input_height   = 32
                self.input_width    = 32
                self.input_nchannels = 3
            elif args.network=='autoencoder':
                self.input_dim = 3072
            self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = args.keep_prob

        ## learning schedule
        self.max_epochs = args.max_epochs
        self.batch_size = 128
        self.early_stopping = 10

        ## optimizer parameters
        self.optimizer     = args.optimizer
        if self.optimizer=='sgd':
            self.learning_rate = args.learning_rate # 1e-1 for sgd, 1e-3 for adam
            self.momentum      = args.momentum
            self.nesterov = args.nesterov
        elif self.optimizer=='adam':
            self.learning_rate = args.learning_rate # 1e-1 for sgd, 1e-3 for adam
            self.beta1         = args.beta1
            self.beta2         = args.beta2
            self.epsilon       = args.epsilon 
        elif self.optimizer=='kalpit':
            self.momentum      = args.momentum
            self.max_lr        = args.max_lr
            self.magic_2nd_order = False
        
    def save(self, save_dir):
        fname = os.path.join(save_dir, 'config.pkl') 
        with open(fname, 'w') as f:
            pickle.dump(self.__dict__, f, 2)
        fname = os.path.join(save_dir, 'config.txt') 
        with open(fname, 'w') as f:
            for k in self.__dict__:
                f.write(k+': '+str(self.__dict__[k])+'\n')


def validate(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
                 model.use_past_bt: False,
                 model.h1_past_bt: np.zeros((dataset.n_val,model.cfg.h1_dim)),
                 model.h2_past_bt: np.zeros((dataset.n_val,model.cfg.h2_dim))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc


def main(dataset_name, network, save_dir, cfg):
    ## Data
    data_dir = os.path.join('/scail/data/group/atlas/kalpit/data', dataset_name)
    dataset = Dataset(data_dir)

    ## Model
    print 'Creating Model...'
    model = get_model(dataset_name+'_'+network, cfg)
    #model.summary()

    ## Train
    print 'Training Model...'
    starttime = time.time()
    if network=='ff':
        if cfg.optimizer=='kalpit':
            train_loss, val_loss, val_acc = train_ff_kalpit(model, dataset, cfg, save_dir)
        else:
            train_loss, val_loss, val_acc = train_ff_vanilla(model, dataset, cfg, save_dir)
    elif network=='conv':
        dataset.data_reshape((cfg.input_height,cfg.input_width,cfg.input_nchannels)) # for both mnist and cifar10
        if cfg.optimizer=='kalpit':
            train_loss, val_loss, val_acc = train_conv_kalpit(model, dataset, cfg, save_dir)
        else:
            train_loss, val_loss, val_acc = train_conv_vanilla(model, dataset, cfg, save_dir)
    elif network=='autoencoder':
        if cfg.optimizer=='kalpit':
            train_loss, val_loss = train_autoencoder_kalpit(model, dataset, cfg, save_dir)
        else:
            train_loss, val_loss = train_autoencoder_vanilla(model, dataset, cfg, save_dir)
    else:
        raise NotImplementedError
    endtime = time.time()
    #plot_loss(train_loss, save_dir, 'training_cost', 'training_cost')
    #plot_loss(val_loss, save_dir, 'validation_cost', 'validation_cost')

    ## Validate
    print ''
    print 'Final Validation...'
    if network=='ff':
        validate_ff(model, dataset)
    elif network=='conv':
        validate_conv(model, dataset)
    elif network=='autoencoder':
        validate_autoencoder(model, dataset)
    
    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    return min(train_loss)


def ArgumentParser():
    parser = argparse.ArgumentParser()
    ## general
    parser.add_argument('dataset', choices=['mnist', 'cifar10'])
    parser.add_argument('network', choices=['ff', 'conv', 'autoencoder'])
    parser.add_argument('keep_prob', type=float, default=0.5)
    parser.add_argument('optimizer', choices=['sgd', 'adam', 'kalpit'])
    parser.add_argument('--final_run', action='store_true', default=False)

    ## learning schedule
    parser.add_argument('--max_epochs', type=int, default=1)

    ## optimizer parameters 
    parser.add_argument('--learning_rate', type=float, default=1e-3) # general optimizer
    parser.add_argument('--momentum', type=float, default=0.0)       # general optimizer
    # sgd
    parser.add_argument('--nesterov', action='store_true', default=False) # sgd
    # adam
    parser.add_argument('--beta1',   type=float, default=0.99)  # adam
    parser.add_argument('--beta2',   type=float, default=0.999) # adam
    parser.add_argument('--epsilon', type=float, default=1e-8)  # adam
    # adadelta
    parser.add_argument('--rho', type=float, default=0.95)  # rho
    # kalpit
    parser.add_argument('--max_lr', type=float) # max_lr
    
    return parser.parse_args()


if __name__=="__main__":
    ## parse inputs
    args = ArgumentParser()

    ## final_run
    final_run = args.final_run

    ## Config
    cfg = Config(args) # pass parsed inputs here

    ## get save_dir: (redirect stdout, save config) to save_dir
    save_dir = get_save_dir(args.dataset, args.network, cfg)
    if final_run:
        sys.stdout = open(os.path.join(save_dir, 'stdout'), 'w')
    cfg.save(save_dir)

    ## main
    min_loss = main(args.dataset, args.network, save_dir, cfg)
