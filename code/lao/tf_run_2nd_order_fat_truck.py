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
        self.input_dim  = 784
        self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = 0.5

        self.max_epochs = 1
        self.batch_size = 128
        self.momentum = 0.0 ## momentum is being used
        self.eps  = 0.9  # 1 - 1e-2
        self.eps2 = 0.999 # 1 - 1e-2
        self.epsilon = 1e-8
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
    db = [] # d_biased
    gg2 = []
    timestep = 0
    count = 0
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
            if db==[]:
                #db  = [(1-cfg.eps)*grads[i] for i in range(len(grads))]
                d   = [grads[i] for i in range(len(grads))]
                #gg2 = [(1-cfg.eps2)*grads[i]*grads[i] for i in range(len(grads))]
            else:
                #db  = [cfg.eps*db[i] + (1-cfg.eps)*grads[i] for i in range(len(grads))]
                mult1 = cfg.eps*(1-cfg.eps**(timestep-1))/(1-cfg.eps**timestep)
                mult2 = (1-cfg.eps)/(1-cfg.eps**timestep)
                d   = [mult1*d[i] + mult2*grads[i] for i in range(len(grads))]
                #d  = [db[i]/(1-cfg.eps**timestep) for i in range(len(grads))]
                #gg2 = [cfg.eps*gg2[i] + (1-cfg.eps2)*grads[i]*grads[i] for i in range(len(grads))]
            #d  = [db[i]/(1-cfg.eps**timestep) for i in range(len(grads))]
            #gg = [gg2[i]/(1-cfg.eps2**timestep) for i in range(len(grads))]
            #d  = [d[i]/(np.sqrt(gg[i])+cfg.epsilon) for i in range(len(grads))]
                    
            research_fd = {model.h1_W_grad: d[0],    model.h1_b_grad: d[1],
                           model.h2_W_grad: d[2],    model.h2_b_grad: d[3],
                           model.preds_W_grad: d[4], model.preds_b_grad: d[5],
                          }
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_d = np.sum([np.sum(grads[i]*d[i]) for i in range(len(grads))])
            print 'gT_d: ', time.time()-st

            ## set fd to use old binary tensors
            st = time.time()
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: True,
                  model.h1_past_bt: h1_bt,
                  model.h2_past_bt: h2_bt
                 }
            print 'change fd: ', time.time()-st

            ## get f(x+alpha*d)
            st = time.time()
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            fx_plus_ad, grads2 = model.sess.run([model.loss, model.grads], feed_dict=fd)
            print 'fx+: ', time.time()-st

            ## get f(x-alpha*d)
            st = time.time()
            research_fd[model.lr] = 2*alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            #fx_minus_ad = model.sess.run(model.loss, feed_dict=fd)
            fx_minus_ad, grads3 = model.sess.run([model.loss, model.grads], feed_dict=fd)
            print 'fx-: ', time.time()-st

            ## estimate Hd and dT_H_d
            Hd = [grads2[i]/alpha-grads[i]/alpha for i in range(len(grads))]
            dT_H_d = np.sum([np.sum(d[i]*Hd[i]) for i in range(len(grads))])

            ## choose learning rate
            st = time.time()
            dT_H_d_2 = (fx_plus_ad + fx_minus_ad - 2*fx)/(alpha**2)
            Hd_23 = [grads2[i]/(2.0*alpha)-grads3[i]/(2.0*alpha) for i in range(len(grads))]
            dT_H_d_23 = np.sum([np.sum(d[i]*Hd_23[i]) for i in range(len(grads))])
            print 'dT_H_d_2', dT_H_d, dT_H_d_2, dT_H_d_23
            dT_H_d = dT_H_d
            if not cfg.magic_2nd_order:
                if dT_H_d==0.0:
                    max_lr = lr = 0.0
                else:
                    max_lr = 2*gT_d/np.abs(dT_H_d) # this is a magnitude comment
                    lr = max(min(fx/gT_d, np.abs(max_lr)), -np.abs(max_lr))
                    max_lr_epoch.append(max_lr)
                    lr_epoch.append(lr)
            
            else: ## 2nd order magic
                if dT_H_d==0.0:
                    max_lr = lr = 0.0
                else:
                    delta_f = fx
                    if gT_d**2-2*dT_H_d*delta_f >= 0:
                        if gT_d > 0: # choose the smaller of the two
                            max_lr = lr = - (-gT_d + np.sqrt(gT_d**2-2*dT_H_d*delta_f)) / dT_H_d
                        else:
                            max_lr = lr = - (-gT_d - np.sqrt(gT_d**2-2*dT_H_d*delta_f)) / dT_H_d
                    else:
                        max_lr = lr = - (-gT_d/dT_H_d)
            print 'choose lr: ', time.time()-st
            if max_lr==lr:
                count += 1
    
            ## print
            st = time.time()
            if True:
                print ''
                print 'alpha             : ', alpha
                print 'f(x)              : ', fx
                print 'f(x+alpha*d)      : ', fx_plus_ad
                #print 'f(x-alpha*d)      : ', fx_minus_ad
                #print 'f(x+)+f(x-)-2f(x) : ', fx_plus_ad + fx_minus_ad - 2*fx
                print 'estimated (d.T)Hd : ', dT_H_d
                print '(g.T)d            : ', gT_d
                print 'max lr            : ', max_lr
                print 'lr                : ', lr
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if dT_H_d==0.0:
                print 'dT_H_d==0.0, exiting'
                exit()
            ## update step
            # reset to x
            research_fd[model.lr] = -alpha + lr
            model.sess.run(model.change_weights_op, feed_dict=research_fd)

            ## update alpha
            alpha = min(lr/2, 1e-1)
            alpha = 0.1

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
    print '#max_lr==lr: ', count, '/', timestep
    exit()
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
    return min(train_loss)


if __name__=="__main__":
    ## dataset
    dataset_name = 'mnist'

    ## gpu_run?
    final_run = True

    # adadelta
    #lr = [1e-4 ,1e-3, 1e-2, 1e-1]
    #lr = [1e-3, 1e-2, 1e-1]
    #rho = [0.01, 0.03, 0.1, 0.2]
    #rho = [0.2, 0.3, 0.4, 0.5]
    #params = list(itertools.product(lr, rho))

    # sgd nesterov
    #lr = [1e-4 ,1e-3, 1e-2, 1e-1]
    #mom = [0.9, 0.95, 0.99]
    #params = list(itertools.product(lr, mom))

    # adam
    #lr = [1e-4, 1e-3, 1e-2, 1e-1]
    #beta1 = [0.9, 0.95, 0.99, 0.995]
    #beta2 = [0.9, 0.99, 0.999, 0.9999]
    #params = list(itertools.product(lr, beta1, beta2))
    #params = [(0.0001, 0.9, 0.99)] + params[19:]

    # custom
    eps = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    #eps = [0.0]
    eps = [0.7] # 0.5 to 0.8 were all good
    params = list(itertools.product(eps))


    if params==[]:
        params = ['default']
    print params

    best_loss = float('inf')
    best_run_num = None
    best_params = None

    for i in range(len(params)):
        tf.reset_default_graph()
        print 'now running params: ', params[i]

        ## Config
        cfg = Config()
        # custom
        cfg.eps = params[i][0]
        #cfg.max_epochs = 200

        # sgd
        #cfg.optimizer = 'sgd'
        #cfg.max_epochs = 200
        #cfg.nesterov = True
        #cfg.learning_rate = params[i][0] # 1e-1 for sgd
        #cfg.mom = params[i][1] # for sgd

        # adadelta
        #cfg.optimizer = 'adadelta'
        #cfg.learning_rate = params[i][0]
        #cfg.rho = params[i][1]

        # adam
        #cfg.optimizer = 'adam'
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
