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
from tensorflow.python.ops import math_ops

from data_handler import Dataset
from common.models import get_model
from common.utils import save_loss, plot_loss

DTYPE = 'float32'

class Config(object):
    def __init__(self, save_dir=None):
        self.input_dim = 784
        self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = 0.8

        self.max_epochs = 200
        self.batch_size = 128
        self.momentum = 0.0 ## momentum is being used
        self.use_nesterov = False
        self.optimizer = 'kalpit'
        self.magic_2nd_order = True
        self.acc_rho = 0.0
        self.learning_rate = 1e-1
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


def train(model, dataset, cfg):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss([], save_dir, 'max_learning_rates.txt', first_use=True)
    save_loss([], save_dir, 'learning_rates.txt', first_use=True)
    alpha = 1e-2
    moms = []
    converged = False
    timestep = 0
    ### accumulators
    f_acc = 0.0 # function value accumulator
    u_acc = 0.0 # update accumulator
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        est = time.time()
        for batch_num in range(tot_batches):
            timestep += 1
            bst = time.time()
            st = time.time()
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: False,
                  model.h1_past_bt: np.zeros((len(batch_inds),model.cfg.h1_dim)),
                  model.h2_past_bt: np.zeros((len(batch_inds),model.cfg.h2_dim))
                 }
            loss, acc, grads, h1_bt, h2_bt = model.sess.run([model.loss, model.accuracy, model.grads, 
                                                                 model.h1_binary_tensor, model.h2_binary_tensor], 
                                                                 feed_dict=fd)
            fx = loss
            train_loss_batch.append(loss)
            train_acc_batch.append(acc)
            f_acc = cfg.acc_rho*f_acc + fx
            ### mom
            if moms==[]:
                moms = [np.zeros(g.shape) for g in grads]
            #mult1 =    model.cfg.momentum  * (1-model.cfg.momentum**(timestep-1)) / (1-model.cfg.momentum**timestep)
            #mult2 = (1-model.cfg.momentum) / (1-model.cfg.momentum**timestep)
            #moms = [(mult1)*moms[i] + (mult2)*grads[i] for i in range(len(moms))]    
            moms = [model.cfg.momentum*moms[i]+grads[i] for i in range(len(grads))] # vanilla momentum
        
            research_fd = {model.h1_W_grad: moms[0],    model.h1_b_grad: moms[1],
                           model.h2_W_grad: moms[2],    model.h2_b_grad: moms[3],
                           model.preds_W_grad: moms[4],    model.preds_b_grad: moms[5]
                          }
            ### mom end
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_m = np.sum([np.sum(grads[i]*moms[i]) for i in range(len(grads))])
            print 'gT_m: ', time.time()-st

            ## set fd to use old binary tensors
            st = time.time()
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: True,
                  model.h1_past_bt: h1_bt,
                  model.h2_past_bt: h2_bt
                 }
            print 'change fd: ', time.time()-st

            ## get f(x+alpha*m)
            st = time.time()
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            fx_plus_am = model.sess.run(model.loss, feed_dict=fd)
            print 'fx+: ', time.time()-st

            ## get f(x-alpha*m)
            st = time.time()
            research_fd[model.lr] = 2*alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            fx_minus_am = model.sess.run(model.loss, feed_dict=fd)
            print 'fx-: ', time.time()-st

            """
            ## choose learning rate
            st = time.time()
            mT_H_m = (fx_plus_am + fx_minus_am - 2*fx)/(alpha**2)
            if not cfg.magic_2nd_order:
                max_lr = 2*gT_m/np.abs(mT_H_m)
                lr = min(fx/gT_m, max_lr)
            
            else: ## 2nd order magic
                if mT_H_m==0.0:
                    max_lr = lr = 0.0
                else:
                    delta_f = fx
                    if gT_m**2-2*mT_H_m*delta_f >= 0:
                        max_lr = lr = - (-gT_m + np.sqrt(gT_m**2-2*mT_H_m*delta_f)) / mT_H_m
                    else:
                        max_lr = lr = - (-gT_m/mT_H_m)
            """
            ## choose learning rate for fx + beta*d*g + beta**2/2*dT*H*d
            ## note that lr will be -beta
            st = time.time()
            mT_H_m = (fx_plus_am + fx_minus_am - 2*fx)/(alpha**2)
            c = f_acc + u_acc # in ax**2 + bx + c 
            b = 1.0*(1-cfg.acc_rho**timestep)/(1-cfg.acc_rho) * gT_m
            a = 1.0*(1-cfg.acc_rho**timestep)/(1-cfg.acc_rho) * mT_H_m / 2.0
    
            if not cfg.magic_2nd_order:
                max_lr = np.abs(b/a) # 2*gT_m/np.abs(mT_H_m)
                lr = - (-c/b)
                lr = max(min(max_lr, lr), -max_lr)
            
            else: ## 2nd order magic
                if mT_H_m==0.0:
                    max_lr = lr = 0.0
                else:
                    if b**2-4*a*c >= 0:
                        max_lr = lr = - (-b + np.sqrt(b**2-4*a*c)) / (2*a)
                    else:
                        max_lr = lr = - (-b/(2*a))
            
            clip_max_lr = np.abs(b/a)
            lr = max(min(clip_max_lr, lr), -clip_max_lr)
           
            max_lr_epoch.append(max_lr)
            lr_epoch.append(lr)

            print 'choose lr: ', time.time()-st
    
            ## print
            st = time.time()
            if True:
                print ''
                print 'alpha             : ', alpha
                print 'f(x)              : ', fx
                print 'f(x+alpha*m)      : ', fx_plus_am
                print 'f(x-alpha*m)      : ', fx_minus_am
                print 'f(x+)+f(x-)-2f(x) : ', fx_plus_am + fx_minus_am - 2*fx
                print 'estimated (m.T)Hm : ', mT_H_m
                print '(g.T)m            : ', gT_m
                print 'f_acc             : ', f_acc
                print 'u_acc             : ', u_acc
                print 'max lr            : ', max_lr
                print 'lr                : ', lr
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            print 'printing: ', time.time()-st

            ## update accumulation
            u_n = -lr*gT_m + lr**2/2*mT_H_m # update_n
            u_acc += 1.0*(1-cfg.acc_rho**timestep)/(1-cfg.acc_rho) * u_n
            u_acc *= cfg.acc_rho

            ## quit?
            st = time.time()
            if mT_H_m==0.0:
                print 'mT_H_m==0.0, exiting'
                converged = True
                break


            ## update step
            # reset to x
            research_fd[model.lr] = -alpha + lr
            model.sess.run(model.change_weights_op, feed_dict=research_fd)

            ## update alpha
            alpha = min(lr/2, 1e-1)
            alpha = 1e-2

            print 'quit? final update, alpha: ', time.time()-st
            print 'batch_time: ', time.time()-bst
            print '_'*100
        print 'avg_batch_time: ', (time.time()-est)/tot_batches

        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(max_lr_epoch, save_dir, 'max_learning_rates.txt')
        save_loss(lr_epoch, save_dir, 'learning_rates.txt')
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        if converged:
            break
        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc


def validate(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
                 model.use_past_bt: False,
                 model.input_past_bt: np.zeros((len(batch_inds),cfg.input_height,cfg.input_width,cfg.input_nchannels)),
                 model.fc4_past_bt: np.zeros((len(batch_inds),1000))
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
    #dataset.data_reshape((cfg.input_height,cfg.input_width,cfg.input_nchannels))

    ## Model
    print 'Creating Model...'
    model = get_model(dataset_name, cfg)
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
    #validate(model, dataset)

    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
    return min(train_loss)


if __name__=="__main__":
    ## dataset
    dataset_name = 'mnist'

    ## gpu_run?
    final_run = False

    ## adadelta
    learning_rate = [0.1]
    rho = [0.95]
    #acc_rho = [0.0]
    params = list(itertools.product(learning_rate, rho))

    ## kalpit
    #momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #momentum = [0.5]
    #acc_rho = [0.0]
    #params = list(itertools.product(momentum, acc_rho))

    best_loss = float('inf')
    best_run_num = None
    best_params = None

    print params
    for i in range(len(params)):
        tf.reset_default_graph()
        print 'now running params: ', params[i]

        ## Config
        cfg = Config()

        ## adadelta
        cfg.optimizer = 'adadelta'
        cfg.learning_rate = params[i][0]
        cfg.rho = params[i][1]
        cfg.max_epochs = 100
        cfg.keep_prob = 0.5

        ## kalpit
        #cfg.optimizer = 'kalpit'
        #cfg.max_epochs = 30
        #cfg.keep_prob = 0.5
        #cfg.magic_2nd_order = False
        #cfg.momentum = params[i][0] 
        #cfg.acc_rho = params[i][1] 

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


