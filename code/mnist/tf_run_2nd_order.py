import sys
import os
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
from dropout import dropout

DTYPE = 'float32'

class Config(object):
    def __init__(self, save_dir=None):
        self.input_dim  = 784
        self.output_dim = 10
        self.h1_dim     = 1000
        self.h2_dim     = 1000
        self.keep_prob  = 0.5

        self.max_epochs = 100
        self.batch_size = 128
        self.learning_rate = 1e-1
        self.momentum = 0.0
        self.optimizer = 'sgd'
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


class Model(object):
    def __init__(self, config):
        self.cfg = config
        self.create_feedforward_classifier_model()
        self.initialize()

    def create_feedforward_classifier_model(self):
        """
        Creates:
        self.input_images
        self.labels
        self.lr
        self.preds - pre-softmax predictions
        self.loss
        self.accuracy
        self.grads
        self.train_op
        self.change_weights_op
        """
        ## input placeholders
        self.input_images = tf.placeholder(tf.float32, shape=(None,self.cfg.input_dim), name='input_images')
        self.labels       = tf.placeholder(tf.int32,   shape=(None,), name='labels')
        self.lr           = tf.placeholder(tf.float32, shape=(), name='lr')
        self.use_past_bt  = tf.placeholder(tf.bool,    shape=(), name='use_past_bt') # to pass previous dropout mask
        self.h1_past_bt   = tf.placeholder(tf.float32, shape=(None, cfg.h1_dim), name='h1_past_bt')
        self.h2_past_bt   = tf.placeholder(tf.float32, shape=(None, cfg.h2_dim), name='h2_past_bt')

        ## forward pass, note how this is pre-softmax
        h1 = layers.fully_connected(self.input_images, num_outputs=cfg.h1_dim, activation_fn=tf.nn.relu, 
                                    biases_initializer=layers.initializers.xavier_initializer(), scope='h1')
        h1, self.h1_binary_tensor = tf.cond(self.use_past_bt, lambda: [math_ops.div(h1,cfg.keep_prob)*self.h1_past_bt, self.h1_past_bt], 
                                            lambda: dropout(h1, keep_prob=cfg.keep_prob))
        #h1, self.h1_binary_tensor = dropout(h1, keep_prob=cfg.keep_prob)
        #h1 = tf.nn.dropout(h1, keep_prob=cfg.keep_prob)
        h2 = layers.fully_connected(h1, num_outputs=cfg.h2_dim, activation_fn=tf.nn.relu,
                                    biases_initializer=layers.initializers.xavier_initializer(), scope='h2')
        h2, self.h2_binary_tensor = tf.cond(self.use_past_bt, lambda: [math_ops.div(h2,cfg.keep_prob)*self.h2_past_bt, self.h2_past_bt], 
                                            lambda: dropout(h2, keep_prob=cfg.keep_prob))
        #h2, self.h2_binary_tensor = dropout(h2, keep_prob=cfg.keep_prob)
        #h2 = tf.nn.dropout(h2, keep_prob=cfg.keep_prob)
        self.preds = layers.fully_connected(h2, num_outputs=self.cfg.output_dim, activation_fn=None,
                                            biases_initializer=layers.initializers.xavier_initializer(), scope='preds')

        ## loss and accuracy
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.preds, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='loss', axis=None)
        #self.accuracy = tf.contrib.metrics.accuracy(labels=tf.one_hot(self.labels, cfg.output_dim, dtype='float32'), predictions=self.preds)
        self.accuracy = tf.contrib.metrics.accuracy(labels=self.labels, predictions=tf.to_int32(tf.argmax(self.preds, axis=1)))

        ## training op
        if cfg.optimizer=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gvs = optimizer.compute_gradients(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.grads, vrbs = zip(*gvs)
        self.train_op = optimizer.apply_gradients(gvs)
        
        ### op to just apply passed gradients
        #main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print main_vars
        #print [x.get_shape() for x in grads]
        self.h1_W_grad = tf.placeholder(tf.float32, shape=(cfg.input_dim,cfg.h1_dim), name='h1_W_grad')
        self.h1_b_grad = tf.placeholder(tf.float32, shape=(cfg.h1_dim), name='h1_b_grad')
        self.h2_W_grad = tf.placeholder(tf.float32, shape=(cfg.h1_dim,cfg.h2_dim), name='h2_W_grad')
        self.h2_b_grad = tf.placeholder(tf.float32, shape=(cfg.h2_dim), name='h2_b_grad')
        self.preds_W_grad = tf.placeholder(tf.float32, shape=(cfg.h2_dim,cfg.output_dim), name='preds_W_grad')
        self.preds_b_grad = tf.placeholder(tf.float32, shape=(cfg.output_dim), name='preds_b_grad')
        passed_grads = [self.h1_W_grad, self.h1_b_grad, 
                        self.h2_W_grad, self.h2_b_grad,
                        self.preds_W_grad, self.preds_b_grad]
        passed_gvs = zip(passed_grads, vrbs)
        self.change_weights_op = optimizer.apply_gradients(passed_gvs)
    
    def initialize(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
    

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
            research_fd = {model.h1_W_grad: grads[0],    model.h1_b_grad: grads[1],
                           model.h2_W_grad: grads[2],    model.h2_b_grad: grads[3],
                           model.preds_W_grad: grads[4], model.preds_b_grad: grads[5],
                          }
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_g = np.sum([np.sum(np.square(g)) for g in grads])
            print 'gT_g: ', time.time()-st

            ## set fd to use old binary tensors
            st = time.time()
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.use_past_bt: True,
                  model.h1_past_bt: h1_bt,
                  model.h2_past_bt: h2_bt
                 }
            print 'change fd: ', time.time()-st

            ## get f(x+alpha*g)
            st = time.time()
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            fx_plus_ag = model.sess.run(model.loss, feed_dict=fd)
            print 'fx+: ', time.time()-st

            ## get f(x-alpha*g)
            st = time.time()
            research_fd[model.lr] = 2*alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            fx_minus_ag = model.sess.run(model.loss, feed_dict=fd)
            print 'fx-: ', time.time()-st

            ## choose learning rate
            st = time.time()
            gT_H_g = (fx_plus_ag + fx_minus_ag - 2*fx)/(alpha**2)
            max_lr = 2*gT_g/np.abs(gT_H_g)
            lr = min(fx/gT_g, max_lr)
            max_lr_epoch.append(max_lr)
            lr_epoch.append(lr)
            print 'choose lr: ', time.time()-st
            
            ###################################
            ## 2nd order magic
            if gT_H_g <= 0.0:
                max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*fx)) / gT_H_g
            else:
                if gT_g**2-2*gT_H_g*fx >= 0:
                    max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*fx)) / gT_H_g
                else:
                    max_lr = lr = - (-gT_g/gT_H_g)
            ###################################
            
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
                 model.labels: dataset.data['val_labels']
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
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
    np.random.seed(0)
    tf.set_random_seed(0)

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
    data_dir = '/scail/data/group/atlas/kalpit/data/mnist'
    dataset = Dataset(data_dir)

    ## Config
    cfg = Config(save_dir)
    
    ## Model
    print 'Creating Model...'
    model = Model(cfg)
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
