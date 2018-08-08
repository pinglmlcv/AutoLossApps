""" This module implement a toy task: linear regression """
# __Author__ == "Haowen Xu"
# __Data__ == "04-08-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

from dataio.dataset import Dataset
import utils
from models.basic_model import Basic_model

logger = utils.get_logger()

def _log(x):
    y = []
    for xx in x:
        y.append(math.log(xx))
    return y

def _normalize1(x):
    y = []
    for xx in x:
        y.append(1 + math.log(xx + 1e-5) / 12)
    return y

def _normalize2(x):
    y = []
    for xx in x:
        y.append(min(1, xx / 20))
    return y

def _normalize3(x):
    y = []
    for xx in x:
        y.append(xx)
    return y


class Reg(Basic_model):
    def __init__(self, config, sess, exp_name='new_exp', loss_mode='1'):
        super(Reg, self).__init(config, sess,exp_name)
        # ----Loss_mode is only for DEBUG usage.----
        #   0: only mse, 1: mse & l1
        self.update_steps = 0
        self.loss_mode = loss_mode
        self.exp_name = exp_name
        self._load_dataset()
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()
        self.reset()
        self.reward_baseline = None
        self.improve_baseline = None

    def reset(self):
        """ Reset the model """
        # TODO(haowen) The way to carry step number information should be
        # reconsiderd
        self.update_steps = 0
        self.previous_action = [0, 0]
        num_pre = self.config.task.num_pre_loss
        self.previous_mse_loss = deque(maxlen=num_pre)
        self.previous_l1_loss = deque(maxlen=num_pre)
        self.previous_valid_loss = deque(maxlen=num_pre)
        self.previous_train_loss = deque(maxlen=num_pre)
        self.improve_history = deque(maxlen=20)

        # to control when to terminate the episode
        self.endurance = 0
        self.best_loss = 1e10
        self.improve_baseline = None

    def _load_dataset(self):
        conf = self.config.data
        train_c_data_file = conf.train_c_data_file
        valid_c_data_file = conf.valid_c_data_file
        train_t_data_file = conf.train_t_data_file
        valid_t_data_file = conf.valid_t_data_file
        test_data_file = conf.test_data_file
        self.train_c_dataset = Dataset()
        self.train_c_dataset.load_npy(train_c_data_file)
        self.valid_c_dataset = Dataset()
        self.valid_c_dataset.load_npy(valid_c_data_file)
        self.train_t_dataset = Dataset()
        self.train_t_dataset.load_npy(train_t_data_file)
        self.valid_t_dataset = Dataset()
        self.valid_t_dataset.load_npy(valid_t_data_file)
        self.test_dataset = Dataset()
        self.test_dataset.load_npy(test_data_file)

    def _build_placeholder(self):
        x_size = self.config.task.dim_input
        with self.variable_scope('placeholder'):
            self.x_plh = tf.placeholder(shape=[None, x_size], dtype=tf.float32)
            self.y_plh = tf.placeholder(shape=[None], dtype=tf.float32)

    def _build_graph(self):
        config = self.config.task
        h_size = config.dim_hidden
        y_size = config.dim_output
        lr = config.lr

        with self.variable_scope('quadratic'):
            # ----quadratic equation----
            #  ---first order---
            x_size = config.dim_input
            initial = tf.random_normal(shape=[x_size, 1], stddev=0.1, seed=1)
            w1 = tf.Variable(initial)
            sum1 = tf.matmul(self.x_plh, w1)

            #  ---second order---
            initial = tf.random_normal(shape=[x_size, x_size], stddev=0.01,
                                       seed=1)
            w2 = tf.Variable(initial)
            xx = tf.matmul(tf.reshape(self.x_plh, [-1, x_size, 1]),
                           tf.reshape(self.x_plh, [-1, 1, x_size]))
            sum2 = tf.matmul(tf.reshape(xx, [-1, x_size*x_size]),
                             tf.reshape(w2, [x_size*x_size, 1]))
            # NOTE(Haowen): Divide by 10 is important here to promise
            # convergence.
            self.pred = sum1 + sum2
            self.w1 = w1
            self.w2 = w2

            # define loss
            self.loss_mse = tf.reduce_mean(tf.square(tf.squeeze(self.pred)
                                                     - self.y_plh))

            tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=self.exp_name)
            l1_regularizer = tf.contrib.layers.l1_regularizer(
                scale=config.lambda1, scope=None)
            self.loss_l1 = tf.contrib.layers.apply_regularization(
                l1_regularizer, tvars)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=config.lambda2, scope=None)
            self.loss_l2 = tf.contrib.layers.apply_regularization(
                l2_regularizer, tvars)
            if self.loss_mode == '0':
                self.loss_total = self.loss_mse
                print('mse loss')
            elif self.loss_mode == '1':
                self.loss_total = self.loss_mse + self.loss_l1
                print('mse loss and l1 loss')
                print('lambda1:', config.lambda1)
            elif self.loss_mode == '2':
                self.loss_total = self.loss_mse + self.loss_l1 + self.loss_l2
                print('mse loss, l1 loss and l2 loss')
                print('lambda1: {}, lambda2: {}'.format(config.lambda1,
                                                        config.lambda2))
            else:
                raise NotImplementedError

            # ----Define update operation.----
            self.update_mse = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_mse)
            self.update_l1 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l1)
            self.update_l2 = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_l2)
            self.update_total = tf.train.GradientDescentOptimizer(lr).\
                minimize(self.loss_total)
            self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=self.exp_name)
            self.init = tf.global_variables_initializer(self.tvars)

    def update_total(self):
        data = self.train_t_dataset.next_batch(self.config.task.batch_size)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        loss, _ = self.sess.run([self.loss_total, self.update_total],
                                feed_dict=feed_dict)
        return loss

    def valid(self, dataset=None):
        """ test on given dataset """
        if not dataset:
            dataset = self.valid_t_dataset
        data = dataset.next_batch(dataset.num_examples)
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.pred, self.y_plh]
        loss_mse, pred, gdth = self.sess.run(fetch, feed_dict=feed_dict)
        return loss_mse, pred, gdth

    def response(self, action, mode='TRAIN'):
        """ Given an action, return the new state, reward and whether dead

        Args:
            action: one hot encoding of actions

        Returns:
            state: shape = [dim_state_rl]
            reward: shape = [1]
            dead: boolean
        """
        if mode == 'TRAIN':
            dataset = self.train_c_dataset
            dataset_v = self.valid_c_dataset
        else:
            dataset = self.train_t_dataset
            dataset_v = self.valid_t_dataset

        data = dataset.next_batch(self.config.task.batch_size)
        sess = self.sess
        x = data['input']
        y = data['target']
        feed_dict = {self.x_plh: x, self.y_plh: y}
        fetch = [self.loss_mse, self.loss_l1]

        if action == 0:
            # ----Update mse loss.----
            sess.run(self.update_mse, feed_dict=feed_dict)
        elif action == 1:
            # ----Update l1 loss.----
            sess.run(self.update_l1, feed_dict=feed_dict)
        elif action == 2:
            # ----Update l2 loss.----
            sess.run(self.update_l2, feed_dict=feed_dict)

        loss_mse, loss_l1 = sess.run(fetch, feed_dict=feed_dict)
        valid_loss, _, _ = self.valid(dataset=dataset_v)
        train_loss, _, _ = self.valid(dataset=dataset)

        # ----Update state.----
        self.previous_mse_loss = self.previous_mse_loss[1:] + [loss_mse.tolist()]
        self.previous_l1_loss = self.previous_l1_loss[1:] + [loss_l1.tolist()]
        self.previous_action = action.tolist()
        self.update_steps += 1
        self.previous_valid_loss = self.previous_valid_loss[1:]\
            + [valid_loss.tolist()]
        self.previous_train_loss = self.previous_train_loss[1:]\
            + [train_loss.tolist()]

        reward = self.get_step_reward()
        # ----Early stop and record best result.----
        dead = self.check_terminate()
        state = self.get_state()
        return state, reward, dead

    def check_terminate(self):
        # TODO(haowen)
        # Episode terminates on two condition:
        # 1) Convergence: valid loss doesn't improve in endurance steps
        # 2) Collapse: action space collapse to one action (not implement yet)
        step = self.update_steps
        if step % self.config.task.valid_frequency == 0:
            self.endurance += 1
            loss, _, _ = self.valid()
            if loss < self.best_loss:
                self.best_step = self.update_steps
                self.best_loss = loss
                self.endurance = 0
            if self.endurance > self.config.task.max_endurance:
                return True
        return False

    def get_step_reward(self):
        # TODO(haowen) Use the decrease of validation loss as step reward
        if self.improve_baseline is None:
            # ----First step, nothing to compare with.----
            improve = 0.1
        else:
            improve = (self.previous_valid_loss[-2] - self.previous_valid_loss[-1])

        self.improve_history.append(improve)
        self.improve_baseline = mean(self.improve_history)

        #TODO(haowen) Remove nonlinearity
        value = math.sqrt(abs(improve) / (abs(self.improve_baseline) + 1e-5))
        #value = abs(improve) / (abs(self.improve_baseline) + 1e-5)
        value = min(value, self.config.meta.reward_max_value)
        return math.copysign(value, improve)

    def get_final_reward(self):
        assert self.best_loss < 1e10 - 1
        loss_mse = self.best_loss
        reward = self.config.meta.reward_c / loss_mse

        if self.reward_baseline is None:
            self.reward_baseline = reward
        decay = self.config.meta.reward_baseline_decay
        adv = reward - self.reward_baseline
        adv = min(adv, self.config.meta.reward_max_value)
        adv = max(adv, -self.config.meta.reward_max_value)
        # TODO(haowen) Try to use maximum instead of shift average as baseline
        # Result: doesn't seem to help too much
        # ----Shift average----
        self.reward_baseline = decay * self.reward_baseline\
            + (1 - decay) * reward
        # ----Maximun----
        #if self.reward_baseline < reward:
        #    self.reward_baseline = reward
        return reward, adv

    def get_state(self):
        abs_diff = []
        rel_diff = []
        if self.improve_baseline is None:
            ib = 1
        else:
            ib = self.improve_baseline

        for v, t in zip(self.previous_valid_loss, self.previous_train_loss):
            abs_diff.append(v - t)
            if t > 1e-6:
                rel_diff.append(v / t)
            else:
                rel_diff.append(1)

        state = ([math.log(rel_diff[-1])] +
                 _normalize1([abs(ib)]) +
                 _normalize2(self.previous_mse_loss[-1:]) +
                 self.previous_l1_loss[-1:]
                 )
        return np.array(state, dtype='f')

