import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import math_ops

from data_handler import Dataset
from common.models import get_model
from common.utils import save_loss, plot_loss

DTYPE = 'float32'

class Config(object):
    def __init__(self, save_dir=None):
        self.input_dim  = 784
        self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = 0.5

        self.max_epochs = 50
        self.batch_size = 128
        self.momentum = 0.0 ## momentum is being used
        self.use_nesterov = False
        self.optimizer = 'kalpit'
        self.magic_2nd_order = False

        self.notes = 'fx'

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
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        est = time.time()
        for batch_num in range(tot_batches):
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
            ## momentum
            if moms==[]:
                moms = grads[:]
            else:
                moms = [model.cfg.momentum*moms[i] + grads[i] for i in range(len(moms))]    
            research_fd = {model.h1_W_grad: moms[0],    model.h1_b_grad: moms[1],
                           model.h2_W_grad: moms[2],    model.h2_b_grad: moms[3],
                           model.preds_W_grad: moms[4], model.preds_b_grad: moms[5],
                          }
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_m = np.sum([np.sum(grads[i]*moms[i]) for i in range(len(moms))])
            print 'mT_m: ', time.time()-st

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
            fx_plus_am, grads2 = model.sess.run([model.loss, model.grads], feed_dict=fd)
            print 'fx+: ', time.time()-st

            ### get f(x-alpha*g)
            #st = time.time()
            #research_fd[model.lr] = 2*alpha
            #model.sess.run(model.change_weights_op, feed_dict=research_fd)
            #fx_minus_am = model.sess.run(model.loss, feed_dict=fd)
            #print 'fx-: ', time.time()-st

            ## choose learning rate
            st = time.time()
            #mT_H_m = (fx_plus_am + fx_minus_am - 2*fx)/(alpha**2)
            H_m = [grads2[i]/alpha-grads[i]/alpha for i in range(len(grads))]
            mT_H_m = np.sum([np.sum(moms[i]*H_m[i]) for i in range(len(grads))])
            if not cfg.magic_2nd_order:
                max_lr = 2*gT_m/np.abs(mT_H_m)
                lr = min(fx/gT_m, max_lr)
                max_lr_epoch.append(max_lr)
                lr_epoch.append(lr)
            
            else: ## 2nd order magic
                if mT_H_m==0.0:
                    max_lr = lr = 0.0
                else:
                    delta_f = fx
                    if gT_m**2-2*mT_H_m*delta_f >= 0:
                        max_lr = lr = - (-gT_m + np.sqrt(gT_m**2-2*mT_H_m*delta_f)) / mT_H_m
                    else:
                        max_lr = lr = - (-gT_m/mT_H_m)
            print 'choose lr: ', time.time()-st
    
            ## print
            st = time.time()
            if True:
                print ''
                print 'alpha             : ', alpha
                print 'f(x)              : ', fx
                print 'f(x+alpha*m)      : ', fx_plus_am
                #print 'f(x-alpha*m)      : ', fx_minus_am
                #print 'f(x+)+f(x-)-2f(x) : ', fx_plus_am + fx_minus_am - 2*fx
                print 'estimated (m.T)Hm : ', mT_H_m
                print '(g.T)m            : ', gT_m
                print 'max lr            : ', max_lr
                print 'lr                : ', lr
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if mT_H_m==0.0:
                print 'mT_H_m==0.0, exiting'
                exit()

            ## update step
            # reset to x
            research_fd[model.lr] = -alpha + lr
            model.sess.run(model.change_weights_op, feed_dict=research_fd)

            ## update alpha
            alpha = min(lr/2, 1e-1)

            print 'quit? final update, alpha: ', time.time()-st
            print 'batch_time: ', time.time()-bst
            print '_'*100
        print 'avg_batch_time: ', (time.time()-est)/tot_batches

        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(max_lr_epoch, save_dir, 'max_learning_rates.txt')
        save_loss(lr_epoch, save_dir, 'learning_rates.txt')
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc


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
             

if __name__=="__main__":
    ## dataset_name
    dataset_name = 'mnist'

    ## gpu_run?
    final_run = True

    ## create unique run_id and related directory
    while True:
        run_id = np.random.randint(low=1000000000, high=9999999999) # 10 digits
        save_dir = os.path.join(os.getcwd(), 'output_'+str(run_id))
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
    data_dir = os.path.join('/scail/data/group/atlas/kalpit/data', dataset_name)
    dataset = Dataset(data_dir)

    ## Config
    cfg = Config(save_dir)
    
    ## Model
    print 'Creating Model...'
    model = get_model(dataset_name, cfg)
    #model.summary()

    ## Train
    print 'Training Model...'
    starttime = time.time()
    train_loss_batch, train_acc_batch, train_loss, val_loss, val_acc = train(model, dataset, cfg)
    endtime = time.time()
    plot_loss(train_loss, save_dir, 'training_cost', 'training_cost')
    plot_loss(val_loss, save_dir, 'validation_cost', 'validation_cost')

    ## Validate
    print ''
    print 'Final Validation...'
    validate(model, dataset)
    
    ## Training Time
    print 'Training Time: {:.2f}'.format(endtime - starttime)
