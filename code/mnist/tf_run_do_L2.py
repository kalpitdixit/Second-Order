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
        self.input_dim  = 784
        self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = 0.5

        self.max_epochs = 0
        self.batch_size = 128
        self.learning_rate = 1e-1 # 1e-1 for sgd, 1e-3 for adam
        self.beta1 = 0.99  # for adam
        self.beta2 = 0.999 # for adam
        self.momentum = 0.9
        self.nesterov = True
        self.early_stopping = 10
        self.optimizer = 'sgd'
        
    def save(self, save_dir):
        fname = os.path.join(save_dir, 'config.pkl') 
        with open(fname, 'w') as f:
            pickle.dump(self.__dict__, f, 2)
        fname = os.path.join(save_dir, 'config.txt') 
        with open(fname, 'w') as f:
            for k in self.__dict__:
                f.write(k+': '+str(self.__dict__[k])+'\n')


def check_train_perf(model, dataset):
    feed_dict = {model.input_images: dataset.data['train_images'],
                 model.labels: dataset.data['train_labels'],
                 model.keep_prob: 1.0,
                 model.use_past_bt: False,
                 model.h1_past_bt: np.zeros((dataset.n_train,model.cfg.h1_dim)),
                 model.h2_past_bt: np.zeros((dataset.n_train,model.cfg.h2_dim))
                }
    train_loss, train_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'train_loss: {:.3f}  training_acc: {:.3f}\n'.format(train_loss,train_acc)
    return train_loss, train_acc


class L2_manage(object):
    def __init__(self, input_dim, output_dim, out_type):
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.out_type = out_type
        self.scale = 4.0
        # implying computation will be HW+b and not WH+b
        self.out_acc = np.zeros(self.output_dim)
        self.count = 0
        self.HTH = np.zeros((self.input_dim,self.input_dim))
        self.HTO = np.zeros((self.input_dim,self.output_dim))
    def update(self, h, out, labels=None): # hidden state (m x #inputs) and output (m x #outputs)
        # h - given
        # o - desired
        out = self.get_desired_output(out, labels)
        ht = np.transpose(h)
        self.HTH += np.matmul(ht,h)
        self.HTO += np.matmul(ht,out)
        self.out_acc += np.sum(out, axis=0)
        self.count   += h.shape[0]
    def get_desired_output(self, out, labels):
        if self.out_type=='softmax':
            min_out = np.max(out, axis=1)
            out[range(out.shape[0]), labels] = min_out
            max_out = np.max(out, axis=1)
            out[range(out.shape[0]), labels] = max_out + self.scale
        return out
    def get_W_b(self):
        HtH_inv = np.linalg.pinv(self.HTH)
        W = np.matmul(HtH_inv,self.HTO)
        b = self.out_acc / self.count
        return W, b

def do_L2(model, dataset, cfg):
    ## pre-pref
    check_train_perf(model, dataset)
    ## L2 update management objects
    L2_fc3 = L2_manage(model.cfg.h2_dim, model.cfg.output_dim, out_type='softmax')
    ## do L2
    inds = range(dataset.n_train)
    np.random.shuffle(inds)
    tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
    for batch_num in range(tot_batches):
        batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
        feed_dict = {model.input_images: dataset.data['train_images'][batch_inds,:],
                     model.labels: dataset.data['train_labels'][batch_inds],
                     model.keep_prob: 1.0,
                     model.use_past_bt: False,
                     model.h1_past_bt: np.zeros((len(batch_inds),model.cfg.h1_dim)),
                     model.h2_past_bt: np.zeros((len(batch_inds),model.cfg.h2_dim))
                    }
        h2, preds = model.sess.run([model.h2, model.preds], feed_dict=feed_dict)
        out = preds
        L2_fc3.update(h2, out, dataset.data['train_labels'][batch_inds]) # hidden state (m x #inputs) and output (m x #outputs)
    ## assign
    W, b = L2_fc3.get_W_b()
    research_fd = {model.preds_W_grad: W, model.preds_b_grad: b}
    _ = model.sess.run(model.assign_last_layer, feed_dict=research_fd)
    ## post perf
    check_train_perf(model, dataset)

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
                         model.keep_prob: cfg.keep_prob,
                         model.use_past_bt: False,
                         model.h1_past_bt: np.zeros((len(batch_inds),model.cfg.h1_dim)),
                         model.h2_past_bt: np.zeros((len(batch_inds),model.cfg.h2_dim))
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
                 model.keep_prob: cfg.keep_prob,
                 model.use_past_bt: False,
                 model.h1_past_bt: np.zeros((dataset.n_val,model.cfg.h1_dim)),
                 model.h2_past_bt: np.zeros((dataset.n_val,model.cfg.h2_dim))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc


def get_save_dir(cfg):
    save_dir = os.path.join(os.getcwd(), 'results', cfg.optimizer)
    if os.path.exists(save_dir):
        with open('num_files', 'r') as f:
            nfiles = int(f.readline().strip())
    else:
        nfiles = 0
    with open('num_files', 'w') as f:
        f.write(str(nfiles+1))
    save_dir = os.path.join(save_dir, 'run_'+str(nfiles+1))
    os.makedirs(save_dir)
    print 'save_dir: ', save_dir
    return save_dir


def main(dataset_name, save_dir, cfg):
    ## Data
    data_dir = os.path.join('/scail/data/group/atlas/kalpit/data', dataset_name)
    dataset = Dataset(data_dir)

    ## Model
    print 'Creating Model...'
    model = get_model(dataset_name, cfg)
    #model.summary()

    ## Train
    print 'Training Model...'
    starttime = time.time()
    train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc = train(model, dataset, cfg)
    do_L2(model, dataset, cfg)
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
    final_run = False

    #lr = [1e-4, 1e-3, 1e-2, 1e-1]
    #mom = [0.9, 0.95, 0.99]
    #params = list(itertools.product(lr, mom))
    #lr = [1e-3]
    #beta1 = [0.99]
    #beta2 = [0.999]
    #params = list(itertools.product(lr, beta1, beta2))

    best_loss = float('inf')
    best_run_num = None
    best_params = None
    params = ['sgd L2 last']

    for i in range(len(params)):
        tf.reset_default_graph()
        print 'now running params: ', params[i]

        ## Config
        cfg = Config()
        #cfg.optimizer = 'sgd'
        #cfg.nesterov = True
        #cfg.learning_rate = params[i][0] # 1e-1 for sgd
        #cfg.mom = params[i][1] # for sgd
        #cfg.beta1 = params[i][1] # for sgd
        #cfg.beta2 = params[i][2] # for sgd

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
