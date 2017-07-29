import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from common.utils import save_loss, plot_loss, early_stopping


def get_dixit_lr(loss, grads, moms, cfg):
    fx = loss
    gT_g = np.sum([np.sum(np.square(g)) for g in grads])
    gT_d = np.sum([np.sum(grads[i]*moms[i]) for i in range(len(grads))])
    dT_d = np.sum([np.sum(np.square(m)) for m in moms])
    #lr = fx/gT_d
    lr = fx/np.sqrt(gT_g)/np.sqrt(dT_d)
    
    print 'USING cfg.max_lr'
    lr = min(lr,cfg.max_lr)


    ## print
    if True:
        print ''
        print 'f(x)              : ', fx
        print '(g.T)g            : ', gT_g
        print '(g.T)d            : ', gT_d
        print '(d.T)d            : ', dT_d
        print 'lr                : ', lr
        print 'update-size       : ', np.sum([np.sum(np.square(lr*m)) for m in moms])
    return lr, lr # max_lr, lr


def get_kalpit_lr(model, cfg, research_fd, fd, loss, alpha, grads):
    fx = loss
    gT_g = np.sum([np.sum(np.square(g)) for g in grads])

    ## get f(x+alpha*g)
    research_fd[model.lr] = -alpha
    model.sess.run(model.change_weights_op, feed_dict=research_fd)
    fx_plus_ag = model.sess.run(model.loss, feed_dict=fd)

    ## get f(x-alpha*g)
    research_fd[model.lr] = 2*alpha
    model.sess.run(model.change_weights_op, feed_dict=research_fd)
    fx_minus_ag = model.sess.run(model.loss, feed_dict=fd)

    ## choose learning rate
    gT_H_g = (fx_plus_ag + fx_minus_ag - 2*fx)/(alpha**2)
    if not cfg.magic_2nd_order:
        max_lr = 2*gT_g/np.abs(gT_H_g)
        lr = min(fx/gT_g, max_lr)

    else: ## 2nd order magic
        if gT_H_g==0.0:
            max_lr = lr = 0.0
        else:
            delta_f = fx
            if gT_g**2-2*gT_H_g*delta_f >= 0:
                max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*delta_f)) / gT_H_g
            else:
                max_lr = lr = - (-gT_g/gT_H_g)

    ## print
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

    ## quit?
    if gT_H_g==0.0:
        print 'gT_H_g==0.0, exiting'
        exit()

    # reset to x
    research_fd[model.lr] = -alpha
    model.sess.run(model.change_weights_op, feed_dict=research_fd)
    return max_lr, lr

def train_ff_vanilla(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss(val_acc, save_dir, 'validation_accuracy.txt', first_use=True)
    time_since_improvement = 0 # early stopping
    train_time = 0.0
    for epoch in range(cfg.max_epochs):
        st = time.time()
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

        ## train time
        train_time += time.time()-st
        print 'Total Train Time:', train_time

        ## validation
        vl, va = validate_ff(model, dataset)
        val_loss.append(vl)
        val_acc.append(va)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
        save_loss(val_acc[-1:], save_dir, 'validation_accuracy.txt')   

        ## early stopping
        time_since_improvement, early_stop = early_stopping(val_loss, time_since_improvement, cfg.early_stopping)
        if early_stop:
            break
            
    return train_loss, val_loss, val_acc

 
def train_ff_kalpit(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss(val_acc, save_dir, 'validation_accuracy.txt', first_use=True)
    save_loss([], save_dir, 'max_learning_rates.txt', first_use=True)
    save_loss([], save_dir, 'learning_rates.txt', first_use=True)
    count1 = 0
    count2 = 0
    time_since_improvement = 0 # early stopping
    train_time = 0.0
    for epoch in range(cfg.max_epochs):
        st = time.time()
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        est = time.time()
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.keep_prob: cfg.keep_prob,
                  model.use_past_bt: False,
                  model.h1_past_bt: np.zeros((len(batch_inds),model.cfg.h1_dim)),
                  model.h2_past_bt: np.zeros((len(batch_inds),model.cfg.h2_dim)),
                  model.max_lr: cfg.max_lr
                 }
            loss, acc, lr, _, gT_d = model.sess.run([model.loss, model.accuracy, model.lr, model.dixit_train_op, model.gT_d],
                                                    feed_dict=fd)
            train_loss_batch.append(loss)
            train_acc_batch.append(acc)
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}  learning_rate:{:.3f}  gT_d:{:.3f}'.format(epoch+1,batch_num+1,
                  train_loss_batch[-1],train_acc_batch[-1],lr, gT_d)

            ## get kalpit learning_rate
            print 'USING DIXIT LR'
            max_lr_epoch.append(cfg.max_lr)
            lr_epoch.append(lr)
            if lr > 0.999*cfg.max_lr:
                count1 += 1.0
            count2 += 1.0
            print '_'*100

        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(max_lr_epoch, save_dir, 'max_learning_rates.txt')
        save_loss(lr_epoch, save_dir, 'learning_rates.txt')
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        print 'Percentage of lr==max_lr: {:.3f}'.format(100.0*count1/count2)

        ## train time
        train_time += time.time()-st
        print 'Total Train Time:', train_time

        ## validation
        vl, va = validate_ff(model, dataset)
        val_loss.append(vl)
        val_acc.append(va)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
        save_loss(val_acc[-1:], save_dir, 'validation_accuracy.txt')

        ## early stopping
        time_since_improvement, early_stop = early_stopping(val_loss, time_since_improvement, cfg.early_stopping)
        if early_stop:
            break
    return train_loss, val_loss, val_acc


def train_conv_vanilla(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss(val_acc, save_dir, 'validation_accuracy.txt', first_use=True)
    time_since_improvement = 0 # early stopping
    train_time = 0.0
    for epoch in range(cfg.max_epochs):
        st = time.time()
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

        ## train time
        train_time += time.time()-st
        print 'Total Train Time:', train_time

        ## validation
        vl, va = validate_conv(model, dataset)
        val_loss.append(vl)
        val_acc.append(va)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
        save_loss(val_acc[-1:], save_dir, 'validation_accuracy.txt')

        ## early stopping
        time_since_improvement, early_stop = early_stopping(val_loss, time_since_improvement, cfg.early_stopping)
        if early_stop:
            break
    return train_loss, val_loss, val_acc


def train_conv_kalpit(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss(val_acc, save_dir, 'validation_accuracy.txt', first_use=True)
    save_loss([], save_dir, 'max_learning_rates.txt', first_use=True)
    save_loss([], save_dir, 'learning_rates.txt', first_use=True)
    count1 = 0
    count2 = 0
    time_since_improvement = 0 # early stopping
    train_time = 0.0
    for epoch in range(cfg.max_epochs):
        st = time.time()
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        count1 = 0.0
        count2 = 0.0
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:],
                  model.labels: dataset.data['train_labels'][batch_inds],
                  model.keep_prob: cfg.keep_prob,
                  model.use_past_bt: False,
                  model.input_past_bt: np.zeros((len(batch_inds),cfg.input_height,cfg.input_width,cfg.input_nchannels)),
                  model.fc4_past_bt: np.zeros((len(batch_inds),1000)),
                  model.max_lr: cfg.max_lr
                 }
            loss, acc, lr, _, gT_d = model.sess.run([model.loss, model.accuracy, model.lr, model.dixit_train_op, model.gT_d],
                                                    feed_dict=fd)
            train_loss_batch.append(loss)
            train_acc_batch.append(acc)
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}  train_acc:{:.3f}  learning_rate:{:.3f}  gT_d:{:.3f}'.format(epoch+1,batch_num+1,
                  train_loss_batch[-1],train_acc_batch[-1],lr, gT_d)
            ## get kalpit learning_rate
            print 'USING DIXIT LR'
            max_lr_epoch.append(cfg.max_lr)
            lr_epoch.append(lr)
            if lr > 0.999*cfg.max_lr:
                count1 += 1.0
            count2 += 1.0
            print '_'*100

        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(max_lr_epoch, save_dir, 'max_learning_rates.txt')
        save_loss(lr_epoch, save_dir, 'learning_rates.txt')
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])
        print 'Percentage of lr==max_lr: {:.3f}'.format(100.0*count1/count2)

        ## train time
        train_time += time.time()-st
        print 'Total Train Time:', train_time

        ## validation
        vl, va = validate_conv(model, dataset)
        val_loss.append(vl)
        val_acc.append(va)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
        save_loss(val_acc[-1:], save_dir, 'validation_accuracy.txt')

        ## early stopping
        time_since_improvement, early_stop = early_stopping(val_loss, time_since_improvement, cfg.early_stopping)
        if early_stop:
            break
    return train_loss, val_loss, val_acc


def train_autoencoder_vanilla(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_loss = []
    val_loss = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    time_since_improvement = 0
    print np.min(dataset.data['train_images'])
    print np.max(dataset.data['train_images'])
    print '='*100
    time_since_improvement = 0 # early stopping
    train_time = 0.0
    for epoch in range(cfg.max_epochs):
        st = time.time()
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        for batch_num in range(tot_batches):
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            feed_dict = {model.input_images: dataset.data['train_images'][batch_inds,:],
                         model.lr: cfg.learning_rate,
                        }
            loss, _ = model.sess.run([model.loss, model.train_op], feed_dict=feed_dict)
            train_loss_batch.append(loss)
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}'.format(epoch+1,batch_num+1,train_loss_batch[-1])
        train_loss.append(np.mean(train_loss_batch[-tot_batches:]))
        save_loss(train_loss[-1:], save_dir, 'training_cost.txt')
        print 'Epoch {} - Average Training Cost: {:.3f}'.format(epoch+1, train_loss[-1])

        ## train time
        train_time += time.time()-st
        print 'Total Train Time:', train_time

        ## validation
        vl, va = validate_autoencoder(model, dataset)
        val_loss.append(vl)
        save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')

        ## early stopping
        time_since_improvement, early_stop = early_stopping(val_loss, time_since_improvement, cfg.early_stopping)
        if early_stop:
            break
    return train_loss, val_loss


def train_autoencoder_kalpit(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_loss = []
    val_loss = []
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    save_loss([], save_dir, 'max_learning_rates.txt', first_use=True)
    save_loss([], save_dir, 'learning_rates.txt', first_use=True)
    alpha = 1e-1
    moms = []
    converged = False
    print 'TRAIN_AUTOENCODER_KALPIT has NOT BEEN SETUP WITH DIXIT_LR or even KALPIT_LR'
    exit()
    ### accumulators
    for epoch in range(cfg.max_epochs):
        inds = range(dataset.n_train)
        np.random.shuffle(inds)
        tot_batches = int(np.ceil(1.0*dataset.n_train/cfg.batch_size))
        max_lr_epoch = []
        lr_epoch = []
        est = time.time()
        for batch_num in range(tot_batches):
            alpha = 1e0
            bst = time.time()
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:]}
            loss, grads = model.sess.run([model.loss, model.grads],
                                          feed_dict=fd)
            train_loss_batch.append(loss)
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}'.format(epoch+1,batch_num+1,train_loss_batch[-1])
            ## set research_fd. set fd to use old binary tensors.
            research_fd = {model.enc1_W_grad: grads[0],    model.enc1_b_grad: grads[1],
                           model.enc2_W_grad: grads[2],    model.enc2_b_grad: grads[3],
                           model.dec2_W_grad: grads[4],    model.dec2_b_grad: grads[5],
                           model.dec1_W_grad: grads[6],    model.dec1_b_grad: grads[7]
                          }
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:]}

            ## get kalpit learning_rate
            max_lr, lr = get_kalpit_lr(model, cfg, research_fd, fd, loss, alpha, grads)
            max_lr_epoch.append(max_lr)
            lr_epoch.append(lr)

            ## update step
            # momentum
            if moms==[]:
                moms = grads[:]
            else:
                moms = [model.cfg.momentum*moms[i] + grads[i] for i in range(len(moms))]
            research_fd = {model.enc1_W_grad: moms[0],    model.enc1_b_grad: moms[1],
                           model.enc2_W_grad: moms[2],    model.enc2_b_grad: moms[3],
                           model.dec2_W_grad: moms[4],    model.dec2_b_grad: moms[5],
                           model.dec1_W_grad: moms[6],    model.dec1_b_grad: moms[7]
                          }
            research_fd[model.lr] = lr
            model.sess.run(model.change_weights_op, feed_dict=research_fd)

            ## update alpha
            alpha = min(lr/2, 1e-1)

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
        vl, va = validate_autoencoder(model, dataset)
        val_loss.append(vl)
        save_loss(val_acc[-1:], save_dir, 'validation_accuracy.txt')
    return train_loss, val_loss


def validate_ff(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
                 model.keep_prob: 1.0,
                 model.use_past_bt: False,
                 model.h1_past_bt: np.zeros((dataset.n_val,model.cfg.h1_dim)),
                 model.h2_past_bt: np.zeros((dataset.n_val,model.cfg.h2_dim))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc


def validate_conv(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
                 model.keep_prob: 1.0,
                 model.use_past_bt: False,
                 model.input_past_bt: np.zeros((dataset.n_val,model.cfg.input_height,model.cfg.input_width,model.cfg.input_nchannels)),
                 model.fc4_past_bt: np.zeros((dataset.n_val,1000))
                }
    val_loss, val_acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print 'validation_loss: {:.3f}  validation_acc: {:.3f}\n'.format(val_loss,val_acc)
    return val_loss, val_acc


def validate_autoencoder(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images']}
    val_loss = model.sess.run([model.loss], feed_dict=feed_dict)
    val_loss = val_loss[0] # because model.sess.run returns a list
    print 'validation_loss: {:.3f}\n'.format(val_loss)
    return val_loss
