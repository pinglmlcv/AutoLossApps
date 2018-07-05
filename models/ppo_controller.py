import tensorflow as tf
import numpy as np
import os

from models.basic_model import Basic_model
import utils
logger = utils.get_logger()

class BasePPO(Basic_model):
    def __init__(self, config, sess, exp_name):
        super(BasePPO, self).__init__(config, sess, exp_name)

    def _build_placeholder(self):
        config = self.config.meta

        with tf.variable_scope('placeholder'):
            self.state = tf.placeholder(tf.float32, shape=[None, config.dim_s],
                                        name='state')
            self.action = tf.placeholder(tf.int32, shape=[None],
                                         name='action')
            self.target_value = tf.placeholder(tf.float32, shape=[None],
                                               name='target_value')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _build_graph(self):
        value, _ = self.build_critic_net('value_net')
        pi, pi_param = self.build_actor_net('actor_net', trainable=True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net',
                                                    trainable=False)
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32),
                              self.action], axis=1)
        pi_wrt_a = tf.gather_nd(params=pi, indices=a_indices, name='pi_wrt_a')
        old_pi_wrt_a = tf.gather_nd(params=old_pi, indices=a_indices,
                                    name='old_pi_wrt_a')

        cliprange = self.config.meta.cliprange

        with tf.variable_scope('critic_loss'):
            adv = self.target_value - value
            self.critic_loss = tf.reduce_mean(tf.square(adv))

        # NOTE: Stop passing gradient through adv
        adv = tf.stop_gradient(adv, name='adv_stop_gradient')

        with tf.variable_scope('actor_loss'):
            ratio = pi_wrt_a / old_pi_wrt_a
            pg_losses1 = adv * ratio
            pg_losses2 = adv * tf.clip_by_value(ratio,
                                                1.0 - cliprange,
                                                1.0 + cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses1,
                                                         pg_losses2))

        #self.sync_op = [oldp.assign(p)
        #                for p, oldp in zip(pi_param, old_pi_param)]
        self.sync_op = [tf.assign(oldp, p)
                        for oldp, p in zip(old_pi_param, pi_param)]
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op_actor = optimizer.minimize(self.actor_loss)
        self.train_op_critic = optimizer.minimize(self.critic_loss)

        self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope=self.exp_name)
        self.init = tf.variables_initializer(self.tvars)
        self.saver = tf.train.Saver(var_list=self.tvars, max_to_keep=1)

        self.pi = pi
        self.value = value
        self.ratio = ratio
        self.old_pi_param = old_pi_param

    def build_actor_net(self, scope, trainable):
        raise NotImplementedError

    def run_step(self, state, ep):
        state = [state]
        pi = self.sess.run(self.pi, {self.state: state})[0]
        dim_a = self.config.meta.dim_a
        action = np.random.choice(dim_a, 1, p=pi)[0]
        return action, pi

    def sync_net(self):
        self.sess.run(self.sync_op)
        logger.info('meta_target_network synchronized')

    def update(self, transition_batch, fi=0):
        state = transition_batch['state']
        action = transition_batch['action']
        target_value = transition_batch['reward']
        fetch = [self.train_op_actor, self.train_op_critic, self.value]
        feed_dict = {self.state: state,
                     self.action: action,
                     self.target_value: target_value,
                     self.lr: self.config.meta.lr}
        _, _, value = self.sess.run(fetch, feed_dict)
        if fi == 0:
            for i in range(60):
                logger.info('value: {}, target_value: {}'.format(value[i], target_value[i]))


class MlpPPO(BasePPO):
    def __init__(self, config, sess, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dim_h = self.config.meta.dim_h
            dim_a = self.config.meta.dim_a
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.leaky_relu,
                trainable=trainable,
                scope='fc1')

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=dim_a,
                activation_fn=None,
                trainable=trainable,
                scope='fc2')

            output = tf.nn.softmax(logits * self.config.meta.logits_scale)
            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return output, param

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            dim_h = self.config.meta.dim_h
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.leaky_relu,
                scope='fc1')

            value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                scope='fc2')

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))

            return value, param


class LSTMPPO(BasePPO):
    # ----
    # Discrete control actor-actor model with LSTM
    # ----
    def __init__(self, config, sess, exp_name='LSTMPPO'):
        super(LSTMPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dim_h = self.config.meta_dim_h_lstm
            nlayers = self.config.meta_nlayers_lstm

        pass

    def build_critic_net(self, scope):
        pass
