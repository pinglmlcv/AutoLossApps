""" Decide which loss to update by Reinforce Learning """
# __Author__ == "Haowen Xu"
# __Data__ == "04-07-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import math

import utils
from models.basic_model import Basic_model

logger = utils.get_logger()

class Controller(Basic_model):
    def __init__(self, config, sess, exp_name='new_exp_ctrl'):
        super(Controller, self).__init__(config, sess, exp_name)
        self.init = tf.constant([1])
        #self._build_placeholder()
        #self._build_graph()

    def sync_net(self):
        pass

    def _build_placeholder(self):
        config = self.config
        a = config.dim_action_rl
        s = config.dim_state_rl
        with self.graph.as_default():
            self.state_plh = tf.placeholder(shape=[None, s],
                                            dtype=tf.float32)
            self.reward_plh = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_plh = tf.placeholder(shape=[None, a], dtype=tf.int32)
            self.lr_plh = tf.placeholder(dtype=tf.float32)

    def _build_graph(self):
        config = self.config
        x_size = config.dim_state_rl
        h_size = config.dim_hidden_rl
        a_size = config.dim_action_rl
        lr = self.lr_plh
        with self.graph.as_default():
            model_name = config.controller_model_name
            initializer = tf.contrib.layers.xavier_initializer(uniform=True)
            if model_name == '2layer':
                hidden = slim.fully_connected(self.state_plh, h_size,
                                              weights_initializer=initializer,
                                              activation_fn=tf.nn.relu)
                self.logits = slim.fully_connected(hidden, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits)
            elif model_name == '2layer_logits_clipping':
                hidden = slim.fully_connected(self.state_plh, h_size,
                                              weights_initializer=initializer,
                                              activation_fn=tf.nn.relu)
                self.logits = slim.fully_connected(hidden, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits /
                                            config.logit_clipping_c)
            elif model_name == 'linear':
                self.logits = slim.fully_connected(self.state_plh, a_size,
                                                   weights_initializer=initializer,
                                                   activation_fn=None)
                self.output = tf.nn.softmax(self.logits)
            elif model_name == 'linear_logits_clipping':
                #self.logits = slim.fully_connected(self.state_plh, a_size,
                #                                   weights_initializer=initializer,
                #                                   activation_fn=None)
                # ----Old version----
                w = tf.get_variable('w', shape=[x_size, a_size], dtype=tf.float32,
                                    initializer=initializer)
                b = tf.get_variable('b', shape=[a_size], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                self.logits = tf.matmul(self.state_plh, w) + b
                self.output = tf.nn.softmax(self.logits /
                                            config.logit_clipping_c)
            else:
                raise Exception('Invalid controller_model_name')

            self.chosen_action = tf.argmax(self.output, 1)
            self.action = tf.cast(tf.argmax(self.action_plh, 1), tf.int32)
            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + self.action
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)
                                        * self.reward_plh)

            # ----Restore gradients and update them after several iterals.----
            self.tvars = tf.trainable_variables()
            tvars = self.tvars
            self.gradient_plhs = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_plh')
                self.gradient_plhs.append(placeholder)

            self.grads = tf.gradients(self.loss, tvars)
            if config.optimizer_ctrl == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif config.optimizer_ctrl == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            self.train_op = optimizer.apply_gradients(zip(self.gradient_plhs, tvars))
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def run_step(self, state, ep, epsilon=0):
        return ep % 5, ep % 5
        #
        # Sample an action from a given state, probabilistically

        # Args:
        #     state: shape = [dim_state_rl]
        #     epsilon: epsilon, exploring rate

        # Returns:
        #     action: shape = [dim_action_rl]
        #
        #lesson_period = [20, 20, 10, 10, 20, 10, 10, 50]
        #lesson_id     = [0,  1,  3,  4,  2,  3,  4,  2 ]
        #lesson_period = [100, 40, 30]
        #lesson_id     = [0, 2, 1]
        lesson_period = [30]
        lesson_id     = [4]
        #lesson_period = [15, 15] + [1] * 3 * 10 + [40]
        #lesson_id = [i for i in range(2)] + [i for i in range(2, 5)] * 10 + [2]
        #lesson_period = [1]*5*100
        #lesson_id = [i for i in range(5)] * 100

        # Ten agents
        #lesson_period = [30] * 9 + [10] * 10 * 8 + [100]
        #lesson_id = [i for i in range(9)] + [i for i in range(9, 19)] * 8 + [9]
        grade = 0
        while ep > lesson_period[grade]:
            ep -= lesson_period[grade]
            grade = min(grade + 1, len(lesson_period)-1)
        return lesson_id[grade], [0, 0]

    def update(self, batch):
        pass

    def get_value(self, state):
        return 0
