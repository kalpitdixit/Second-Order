import pickle

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import math_ops

from dropout import dropout


class MNIST_Model(object):
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
        self.h1_past_bt   = tf.placeholder(tf.float32, shape=(None, self.cfg.h1_dim), name='h1_past_bt')
        self.h2_past_bt   = tf.placeholder(tf.float32, shape=(None, self.cfg.h2_dim), name='h2_past_bt')

        ## forward pass, note how this is pre-softmax
        h1 = layers.fully_connected(self.input_images, num_outputs=self.cfg.h1_dim, activation_fn=tf.nn.relu,
                                    biases_initializer=layers.initializers.xavier_initializer(), scope='h1')
        h1, self.h1_binary_tensor = tf.cond(self.use_past_bt, lambda: [math_ops.div(h1,self.cfg.keep_prob)*self.h1_past_bt, self.h1_past_bt],
                                            lambda: dropout(h1, keep_prob=self.cfg.keep_prob))
        h2 = layers.fully_connected(h1, num_outputs=self.cfg.h2_dim, activation_fn=tf.nn.relu,
                                    biases_initializer=layers.initializers.xavier_initializer(), scope='h2')
        h2, self.h2_binary_tensor = tf.cond(self.use_past_bt, lambda: [math_ops.div(h2,self.cfg.keep_prob)*self.h2_past_bt, self.h2_past_bt],
                                            lambda: dropout(h2, keep_prob=self.cfg.keep_prob))
        self.preds = layers.fully_connected(h2, num_outputs=self.cfg.output_dim, activation_fn=None,
                                            biases_initializer=layers.initializers.xavier_initializer(), scope='preds')

        ## loss and accuracy
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.preds, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='loss', axis=None)
        self.accuracy = tf.contrib.metrics.accuracy(labels=self.labels, predictions=tf.to_int32(tf.argmax(self.preds, axis=1)))

        ## training op
        if self.cfg.optimizer=='kalpit':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr) # can set lr every minibatch
        if self.cfg.optimizer=='sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.cfg.learning_rate, momentum=self.cfg.momentum, 
                                                   use_nesterov=self.cfg.nesterov)
        elif self.cfg.optimizer=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=self.cfg.beta1, beta2=self.cfg.beta2)
        elif self.cfg.optimizer=='adadelta':
            optimizer = tf.train.AdadeltaOptimizer()
        gvs = optimizer.compute_gradients(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.grads, vrbs = zip(*gvs)
        self.train_op = optimizer.apply_gradients(gvs)

        ### op to just apply passed gradients
        self.h1_W_grad = tf.placeholder(tf.float32, shape=(self.cfg.input_dim,self.cfg.h1_dim), name='h1_W_grad')
        self.h1_b_grad = tf.placeholder(tf.float32, shape=(self.cfg.h1_dim), name='h1_b_grad')
        self.h2_W_grad = tf.placeholder(tf.float32, shape=(self.cfg.h1_dim,self.cfg.h2_dim), name='h2_W_grad')
        self.h2_b_grad = tf.placeholder(tf.float32, shape=(self.cfg.h2_dim), name='h2_b_grad')
        self.preds_W_grad = tf.placeholder(tf.float32, shape=(self.cfg.h2_dim,self.cfg.output_dim), name='preds_W_grad')
        self.preds_b_grad = tf.placeholder(tf.float32, shape=(self.cfg.output_dim), name='preds_b_grad')
        passed_grads = [self.h1_W_grad, self.h1_b_grad,
                        self.h2_W_grad, self.h2_b_grad,
                        self.preds_W_grad, self.preds_b_grad]
        passed_gvs = zip(passed_grads, vrbs)
        self.change_weights_op = optimizer.apply_gradients(passed_gvs)

    def initialize(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


class CIFAR10_Model(object):
    def __init__(self, config):
        self.cfg = config
        self.create_convnet_classifier_model()
        self.initialize()

    def create_convnet_classifier_model(self):
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
        self.input_images = tf.placeholder(tf.float32, shape=(None,self.cfg.input_height,self.cfg.input_width,self.cfg.input_nchannels), 
                                           name='input_images')
        self.labels       = tf.placeholder(tf.int32,   shape=(None,), name='labels')
        self.lr           = tf.placeholder(tf.float32, shape=(), name='lr')
        self.use_past_bt   = tf.placeholder(tf.bool, shape=(), name='use_past_bt')
        self.input_past_bt = tf.placeholder(tf.float32, shape=(None,self.cfg.input_height,self.cfg.input_width,self.cfg.input_nchannels),
                                            name='input_past_bt') # past binary tensor
        self.fc4_past_bt   = tf.placeholder(tf.float32, shape=(None,1000),
                                            name='input_past_bt') # past binary tensor
        
        ## forward pass, note how this is pre-softmax
        dropout_input_images, self.input_binary_tensor = tf.cond(self.use_past_bt, 
                                               lambda: [math_ops.div(self.input_images,self.cfg.keep_prob)*self.input_past_bt, self.input_past_bt],
                                               lambda: dropout(self.input_images, keep_prob=self.cfg.keep_prob))
        conv1 = layers.convolution2d(dropout_input_images, num_outputs=64, kernel_size=(5,5), stride=(1,1), 
                                     padding='SAME', biases_initializer=layers.initializers.xavier_initializer(), 
                                     activation_fn=tf.nn.relu, scope='conv1')
        pool1 = tf.nn.max_pool(conv1, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        conv2 = layers.convolution2d(pool1, num_outputs=64, kernel_size=(5,5), stride=(1,1), 
                                     padding='SAME', biases_initializer=layers.initializers.xavier_initializer(), 
                                     activation_fn=tf.nn.relu, scope='conv2')
        pool2 = tf.nn.max_pool(conv2, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        conv3 = layers.convolution2d(pool2, num_outputs=128, kernel_size=(5,5), stride=(1,1), 
                                     padding='SAME', biases_initializer=layers.initializers.xavier_initializer(), 
                                     activation_fn=tf.nn.relu, scope='conv3')
        pool3 = tf.nn.max_pool(conv3, ksize=(1,3,3,1), strides=(1,2,2,1), padding='SAME')
        pool3_flat = layers.flatten(pool3)
    
        fc4 = layers.fully_connected(pool3_flat, num_outputs=1000, activation_fn=tf.nn.relu,
                                     biases_initializer=layers.initializers.xavier_initializer(), scope='fc4')
        fc4, self.fc4_binary_tensor = tf.cond(self.use_past_bt, 
                                              lambda: [math_ops.div(fc4,self.cfg.keep_prob)*self.fc4_past_bt, self.fc4_past_bt],
                                              lambda: dropout(fc4, keep_prob=self.cfg.keep_prob))
        fc5 = layers.fully_connected(fc4, num_outputs=self.cfg.output_dim, activation_fn=None,
                                     biases_initializer=layers.initializers.xavier_initializer(), scope='fc5')
        self.preds = fc5

        ## loss and accuracy
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.preds, labels=self.labels)
        self.loss = tf.reduce_mean(loss, name='loss', axis=None)
        self.accuracy = tf.contrib.metrics.accuracy(labels=self.labels, predictions=tf.to_int32(tf.argmax(self.preds, axis=1)))

        ## training op
        if self.cfg.optimizer=='kalpit':
            #optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.cfg.momentum, 
            #                                       use_nesterov=self.cfg.nesterov)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr) # can set lr every minibatch
        if self.cfg.optimizer=='sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.cfg.learning_rate, momentum=self.cfg.momentum, 
                                                   use_nesterov=self.cfg.nesterov)
        elif self.cfg.optimizer=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=self.cfg.beta1, beta2=self.cfg.beta2)
        elif self.cfg.optimizer=='adadelta':
            optimizer = tf.train.AdadeltaOptimizer()
        gvs = optimizer.compute_gradients(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.grads, vrbs = zip(*gvs)
        self.train_op = optimizer.apply_gradients(gvs)

        ### op to just apply passed gradients
        self.conv1_W_grad = tf.placeholder(tf.float32, shape=(5,5,3,64), name='conv1_W_grad')
        self.conv1_b_grad = tf.placeholder(tf.float32, shape=(64), name='conv1_b_grad')
        self.conv2_W_grad = tf.placeholder(tf.float32, shape=(5,5,64,64), name='conv2_W_grad')
        self.conv2_b_grad = tf.placeholder(tf.float32, shape=(64), name='conv2_b_grad')
        self.conv3_W_grad = tf.placeholder(tf.float32, shape=(5,5,64,128), name='conv3_W_grad')
        self.conv3_b_grad = tf.placeholder(tf.float32, shape=(128), name='conv3_b_grad')
        self.fc4_W_grad = tf.placeholder(tf.float32, shape=(2048,1000), name='fc4_W_grad')
        self.fc4_b_grad = tf.placeholder(tf.float32, shape=(1000), name='fc4_b_grad')
        self.fc5_W_grad = tf.placeholder(tf.float32, shape=(1000,self.cfg.output_dim), name='fc5_W_grad')
        self.fc5_b_grad = tf.placeholder(tf.float32, shape=(self.cfg.output_dim), name='fc5_b_grad')

        passed_grads = [self.conv1_W_grad, self.conv1_b_grad,
                        self.conv2_W_grad, self.conv2_b_grad,
                        self.conv3_W_grad, self.conv3_b_grad,
                        self.fc4_W_grad,   self.fc4_b_grad,
                        self.fc5_W_grad,   self.fc5_b_grad]
        passed_gvs = zip(passed_grads, vrbs)
        self.change_weights_op = optimizer.apply_gradients(passed_gvs)
        
    def initialize(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


def get_model(dataset_name, config):
    if dataset_name=='mnist':
        return MNIST_Model(config)
    elif dataset_name=='cifar10':
        return CIFAR10_Model(config)