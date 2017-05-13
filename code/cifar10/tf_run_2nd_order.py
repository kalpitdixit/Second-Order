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

from data_handler import Dataset
from common.models import get_model
from common.utils import save_loss, plot_loss

DTYPE = 'float32'

class Config(object):
    def __init__(self, save_dir=None):
        self.input_height    = 32
        self.input_width     = 32
        self.input_nchannels = 3
        self.output_dim = 10
        self.keep_prob  = 0.8

        self.max_epochs = 200
        self.batch_size = 128
        self.optimizer = 'kalpit'
        self.magic_2nd_order = False

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
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        est = time.time()
        times = [[] for i in range(10)]
        for batch_num in range(tot_batches):
            bst = time.time()
            st = time.time()
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: False,
                  model.input_past_bt: np.zeros((len(batch_inds),model.cfg.input_height,model.cfg.input_width,model.cfg.input_nchannels)),
                  model.fc4_past_bt: np.zeros((len(batch_inds),1000))
                }
            loss, acc, grads, input_bt, fc4_bt = model.sess.run([model.loss, model.accuracy, model.grads, 
                                                                 model.input_binary_tensor, model.fc4_binary_tensor], 
                                                                 feed_dict=fd)
            fx = loss
            train_loss_batch.append(loss)
            train_acc_batch.append(acc)
            research_fd = {model.conv1_W_grad: grads[0],    model.conv1_b_grad: grads[1],
                           model.conv2_W_grad: grads[2],    model.conv2_b_grad: grads[3],
                           model.conv3_W_grad: grads[4],    model.conv3_b_grad: grads[5],
                           model.fc4_W_grad: grads[6],      model.fc4_b_grad: grads[7],
                           model.fc5_W_grad: grads[8],      model.fc5_b_grad: grads[9],
                          }
            times[0].append(time.time()-st)
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_g = np.sum([np.sum(np.square(g)) for g in grads])
            times[1].append(time.time()-st)
            print 'gT_g: ', time.time()-st

            ## set fd to use old binary tensors
            st = time.time()
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: True,
                  model.input_past_bt: input_bt,
                  model.fc4_past_bt: fc4_bt
                }
            times[2].append(time.time()-st)
            print 'change fd: ', time.time()-st

            ## get f(x+alpha*g)
            st = time.time()
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            times[3].append(time.time()-st)
            print 'change_weights_op: ', time.time()-st
            st = time.time()
            fx_plus_ag = model.sess.run(model.loss, feed_dict=fd)
            times[4].append(time.time()-st)
            print 'fx+: ', time.time()-st

            ## get f(x-alpha*g)
            st = time.time()
            research_fd[model.lr] = 2*alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            times[5].append(time.time()-st)
            print 'change_weights_op: ', time.time()-st
            st = time.time()
            fx_minus_ag = model.sess.run(model.loss, feed_dict=fd)
            times[6].append(time.time()-st)
            print 'fx-: ', time.time()-st

            ## choose learning rate
            st = time.time()
            gT_H_g = (fx_plus_ag + fx_minus_ag - 2*fx)/(alpha**2)
            if not cfg.magic_2nd_order:
                max_lr = 2*gT_g/np.abs(gT_H_g)
                lr = min(fx/gT_g, max_lr)
            else: ## 2nd order magic
                if gT_g**2-2*gT_H_g*fx > 0:
                    max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*fx)) / gT_H_g
                else:
                    max_lr = lr = - (-gT_g/gT_H_g)

            max_lr_epoch.append(max_lr)
            lr_epoch.append(lr)
            times[7].append(time.time()-st)
            print 'choose lr: ', time.time()-st
            
            ## print
            st = time.time()
            if True:
                print ''
                print 'alpha             : ', alpha
                print 'f(x)              : ', fx
                print 'f(x+alpha*g)      : ', fx_plus_ag
                print 'f(x-alpha*g)      : ', fx_minus_ag
                print 'f(x+)+f(x-)-2f(x) : ', fx_plus_ag + fx_minus_ag - 2*fx
                print 'estimated (g.T)Hg : ', gT_H_g
                print '(g.T)g            : ', gT_g
                print 'max lr            : ', max_lr
                print 'lr                : ', lr
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1],train_acc_batch[-1])
            times[8].append(time.time()-st)
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if gT_H_g==0.0:
                print 'gT_H_g==0.0, exiting'
                exit()

            ## update step
            research_fd[model.lr] = -alpha+lr
            model.sess.run(model.change_weights_op, feed_dict=research_fd)

            ## update alpha
            alpha = min(lr/2, 1e-1)

            times[9].append(time.time()-st)
            print 'quit? final update, alpha: ', time.time()-st
            print 'batch_time: ', time.time()-bst
            print '_'*100
        print 'avg_batch_time: ', (time.time()-est)/tot_batches
        for i in range(len(times)):
            print 'i: ', i, '   ', np.mean(times[i])
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
                 model.input_past_bt: np.zeros((dataset.n_val,model.cfg.input_height,model.cfg.input_width,model.cfg.input_nchannels)),
                 model.fc4_past_bt: np.zeros((dataset.n_val,1000))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc
             

if __name__=="__main__":
    ## dataset_name
    dataset_name = 'cifar10'

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
    print 'DROPOUT NOT IMPLEMENTED CORRECTLY FOR VALIDATION!!!'
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
