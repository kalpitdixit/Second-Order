import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from common.utils import save_loss, plot_loss

def train_ff_vanilla(model, dataset, cfg, save_dir):
    train_loss_batch = [] # record_loss
    train_acc_batch = [] # record_accuracy
    train_loss = []
    val_loss = []
    val_acc = []
    save_loss(train_loss, save_dir, 'training_cost.txt', first_use=True)
    save_loss(val_loss, save_dir, 'validation_cost.txt', first_use=True)
    time_since_improvement = 0 # early stopping
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

        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss, val_loss, val_acc


def train_ff_kalpit(model, dataset, cfg, save_dir):
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
            if not cfg.magic_2nd_order:
                max_lr = 2*gT_g/np.abs(gT_H_g)
                lr = min(fx/gT_g, max_lr)
                max_lr_epoch.append(max_lr)
                lr_epoch.append(lr)

            else: ## 2nd order magic
                if gT_H_g==0.0:
                    max_lr = lr = 0.0
                else:
                    delta_f = fx
                    if gT_g**2-2*gT_H_g*delta_f >= 0:
                        max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*delta_f)) / gT_H_g
                    else:
                        max_lr = lr = - (-gT_g/gT_H_g)
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
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if gT_H_g==0.0:
                print 'gT_H_g==0.0, exiting'
                exit()

            ## update step
            # reset to x
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            # momentum
            if moms==[]:
                moms = grads[:]
            else:
                moms = [model.cfg.momentum*moms[i] + grads[i] for i in range(len(moms))]
            research_fd = {model.h1_W_grad: moms[0],    model.h1_b_grad: moms[1],
                           model.h2_W_grad: moms[2],    model.h2_b_grad: moms[3],
                           model.preds_W_grad: moms[4], model.preds_b_grad: moms[5],
                          }
            research_fd[model.lr] = lr
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


def train_conv_vanilla(model, dataset, cfg, save_dir):
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

        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #val_acc.append(va)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss, val_loss, val_acc


def train_conv_kalpit(model, dataset, cfg, save_dir):
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
                  model.input_past_bt: np.zeros((len(batch_inds),cfg.input_height,cfg.input_width,cfg.input_nchannels)),
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
                           model.input_past_bt: input_bt,   model.fc4_past_bt: fc4_bt
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
                  model.input_past_bt: input_bt,
                  model.fc4_past_bt: fc4_bt
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
            if not cfg.magic_2nd_order:
                max_lr = 2*gT_g/np.abs(gT_H_g)
                lr = min(fx/gT_g, max_lr)
                max_lr_epoch.append(max_lr)
                lr_epoch.append(lr)

            else: ## 2nd order magic
                if gT_H_g==0.0:
                    max_lr = lr = 0.0
                else:
                    delta_f = fx
                    if gT_g**2-2*gT_H_g*delta_f >= 0:
                        max_lr = lr = - (-gT_g + np.sqrt(gT_g**2-2*gT_H_g*delta_f)) / gT_H_g
                    else:
                        max_lr = lr = - (-gT_g/gT_H_g)
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
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if gT_H_g==0.0:
                print 'gT_H_g==0.0, exiting'
                exit()

            ## update step
            # reset to x
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
            # momentum
            if moms==[]:
                moms = grads[:]
            else:
                moms = [model.cfg.momentum*moms[i] + grads[i] for i in range(len(moms))]
            research_fd = {model.conv1_W_grad: moms[0],    model.conv1_b_grad: moms[1],
                           model.conv2_W_grad: moms[2],    model.conv2_b_grad: moms[3],
                           model.conv3_W_grad: moms[4],    model.conv3_b_grad: moms[5],
                           model.fc4_W_grad: moms[6],      model.fc4_b_grad: moms[7],
                           model.fc5_W_grad: moms[8],      model.fc5_b_grad: moms[9]
                          }
            research_fd[model.lr] = lr
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
    for epoch in range(cfg.max_epochs):
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
        if train_loss[-1] == min(train_loss):
            time_since_improvement  = 0
        else:
            time_since_improvement += 1
            if time_since_improvement >= cfg.early_stopping:
                print 'early stopping. no improvement since ', str(cfg.early_stopping), ' epochs.'
                break

        #vl, va = validate(model, dataset)
        #val_loss.append(vl)
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
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
            st = time.time()
            batch_inds = inds[batch_num*cfg.batch_size:min((batch_num+1)*cfg.batch_size,dataset.n_train)]
            ## get f(x) and gradients
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:]}
            loss, grads = model.sess.run([model.loss, model.grads],
                                          feed_dict=fd)
            fx = loss
            train_loss_batch.append(loss)
            research_fd = {model.enc1_W_grad: grads[0],    model.enc1_b_grad: grads[1],
                           model.enc2_W_grad: grads[2],    model.enc2_b_grad: grads[3],
                           model.dec2_W_grad: grads[4],    model.dec2_b_grad: grads[5],
                           model.dec1_W_grad: grads[6],    model.dec1_b_grad: grads[7]
                          }
            ### mom end
            print 'fx and grads: ', time.time()-st
            st = time.time()
            gT_g = np.sum([np.sum(np.square(g)) for g in grads])
            print 'gT_g: ', time.time()-st

            ## set fd to use old binary tensors
            st = time.time()
            fd = {model.input_images: dataset.data['train_images'][batch_inds,:]}
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

            max_lr_epoch.append(max_lr)
            lr_epoch.append(lr)

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
            print 'Epoch-Batch: {:3d}-{:3d}  train_loss: {:.3f}'.format(epoch+1,batch_num+1,
                                                                                          train_loss_batch[-1])
            print 'printing: ', time.time()-st

            ## quit?
            st = time.time()
            if gT_H_g==0.0:
                print 'gT_H_g==0.0, exiting'
                converged = True
                break

            ## update step
            # reset to x
            research_fd[model.lr] = -alpha
            model.sess.run(model.change_weights_op, feed_dict=research_fd)
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
        #save_loss(val_loss[-1:], save_dir, 'validation_cost.txt')
    return train_loss, val_loss


def validate_ff(model, dataset):
    feed_dict = {model.input_images: dataset.data['val_images'],
                 model.labels: dataset.data['val_labels'],
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
