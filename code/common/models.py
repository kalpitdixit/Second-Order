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
            optimizer = tf.train.AdamOptimizer()
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

def get_model(dataset_name, config):
    if dataset_name=='mnist':
        return MNIST_Model(config)
