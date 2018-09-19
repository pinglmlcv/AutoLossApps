import tensorflow as tf
import numpy as np
import os

from models.basic_model import Basic_model
import utils
logger = utils.get_logger()

class BasePPO(Basic_model):
    def __init__(self, config, sess, exp_name):
        super(BasePPO, self).__init__(config, sess, exp_name)
        self.update_steps = 0

    def _build_placeholder(self):
        '''
        Must include:
            self.state,
            self.action,
            self.reward,
            self.next_value,
            self.target_value,
            self.lr


        '''
        raise NotImplementedError

    def _build_graph(self):
        pi, pi_param = self.build_actor_net('actor_net', trainable=True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net',
                                                    trainable=False)
        pi = pi + 1e-8
        old_pi = old_pi + 1e-8
        value, critic_param = self.build_critic_net('value_net')
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32),
                              self.action], axis=1)
        pi_wrt_a = tf.gather_nd(params=pi, indices=a_indices, name='pi_wrt_a')
        old_pi_wrt_a = tf.gather_nd(params=old_pi, indices=a_indices,
                                    name='old_pi_wrt_a')

        cliprange = self.config.meta.cliprange
        gamma = self.config.agent.gamma

        adv = self.target_value - value
        # NOTE: Stop passing gradient through adv
        adv = tf.stop_gradient(adv, name='critic_adv_stop_gradient')
        with tf.variable_scope('critic_loss'):
            critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x))
                                              for x in critic_param])
            #mse_loss = -tf.reduce_mean(adv * value)
            mse_loss = tf.reduce_mean(tf.square(self.target_value - value))
            reg_param = 0.0
            self.critic_loss = mse_loss + reg_param * critic_reg_loss

        #if self.config.meta.one_step_td:
        #    adv = self.reward + gamma * self.next_value - value
        #else:
        #    adv = self.target_value - value
        adv = self.target_value - value
        # NOTE: Stop passing gradient through adv
        adv = tf.stop_gradient(adv, name='actor_adv_stop_gradient')
        with tf.variable_scope('actor_loss'):
            ratio = pi_wrt_a / old_pi_wrt_a
            pg_losses1 = adv * ratio
            pg_losses2 = adv * tf.clip_by_value(ratio,
                                                1.0 - cliprange,
                                                1.0 + cliprange)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(pi * tf.log(pi), 1))
            actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x))
                                            for x in pi_param])
            pg_loss = -tf.reduce_mean(tf.minimum(pg_losses1, pg_losses2))
            beta = self.config.meta.entropy_bonus_beta
            reg_param = 0.0
            self.actor_loss = pg_loss + beta * entropy_loss +\
                reg_param * actor_reg_loss


        self.sync_op = [tf.assign(oldp, p)
                        for oldp, p in zip(old_pi_param, pi_param)]
        optimizer1= tf.train.AdamOptimizer(self.lr)
        optimizer2 = tf.train.AdamOptimizer(self.lr)
        self.train_op_actor = optimizer1.minimize(self.actor_loss)
        self.train_op_critic = optimizer2.minimize(self.critic_loss)

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

    def run_step(self, state, epsilon=0):
        state = [state]
        dim_a = self.config.agent.dim_a
        pi = self.sess.run(self.pi, {self.state: state})[0]
        # epsilon-greedy
        #A = np.ones(dim_a, dtype=float) * epsilon / dim_a
        #A = pi * (1 - epsilon)
        #A += np.ones(dim_a, dtype=float) * epsilon / dim_a
        A = pi
        action = np.random.choice(dim_a, 1, p=A)[0]
        return action, A

    def sync_net(self):
        self.sess.run(self.sync_op)
        logger.info('{}: target_network synchronized'.format(self.exp_name))

    def update_critic(self, transition_batch, lr=0.001):
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']
        fetch = [self.train_op_critic]
        feed_dict = {self.state: state,
                     self.action: action,
                     self.reward: reward,
                     self.target_value: target_value,
                     self.lr: lr}
        _ = self.sess.run(fetch, feed_dict)
        tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 '{}/value_net/fc_2'.format(self.exp_name))
        #self.print_weights(tvar)
        #if self.update_steps % 50 == 0:
        #    for i in range(5):
        #        print(check_value[i], target_value[i])

    def update_actor(self, transition_batch, lr=0.001):
        self.update_steps += 1
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        target_value = transition_batch['target_value']

        fetch = [self.train_op_actor]
        feed_dict = {self.state: state,
                     self.action: action,
                     self.reward: reward,
                     self.target_value: target_value,
                     self.lr: lr}
        self.sess.run(fetch, feed_dict)

class MlpPPO(BasePPO):
    def __init__(self, config, sess, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def _build_placeholder(self):
        config = self.config.agent
        dim_s = config.dim_s
        with tf.variable_scope('placeholder'):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, dim_s],
                                        name='state')
            self.action = tf.placeholder(tf.int32, shape=[None],
                                         name='action')
            self.reward = tf.placeholder(tf.float32, shape=[None],
                                         name='reward')
            self.next_value = tf.placeholder(tf.float32,
                                             shape=[None],
                                             name='next_value')
            self.target_value = tf.placeholder(tf.float32,
                                               shape=[None],
                                               name='target_value')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dim_h = 20
            dim_a = self.config.agent.dim_a
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.tanh,
                trainable=trainable,
                scope='fc1')

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=dim_a,
                activation_fn=None,
                trainable=trainable,
                scope='fc2')

            output = tf.nn.softmax(logits)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return output, param

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            dim_h = 20
            hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=dim_h,
                activation_fn=tf.nn.tanh,
                scope='fc1')

            value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                scope='fc2')
            value = tf.squeeze(value)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return value, param

    def run_step(self, states, epsilon=0):
        dim_a = self.config.agent.dim_a
        if random.random() < epsilon:
            return random.randint(0, self.config.agent.dim_a - 1), 0
        else:
            pi = self.sess.run(self.pi, {self.state: states})[0]
            action = np.random.choice(dim_a, 1, p=pi)[0]
            return action, pi

    def get_value(self, state):
        feed_dict = {self.state: state}
        value = self.sess.run(self.value, feed_dict)
        return value

class CNNPPO(BasePPO):
    def __init__(self, config, sess, exp_name='CNNPPO'):
        super(CNNPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def build_actor_net(self, scope, trainable):
        x = self.state
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_block'):
                filters = [16, 32, 64, 64]
                kernel_sizes = [1, 1, 3, 4]
                strides = [1, 1, 2, 2]
                for i in range(len(filters)):
                    x = tf.layers.conv2d(inputs=x,
                                         filters=filters[i],
                                         kernel_size=kernel_sizes[i],
                                         strides=(strides[i], strides[i]),
                                         padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         trainable=trainable,
                                         name='conv2d_{}'.format(i))
            x = tf.layers.flatten(inputs=x, name='flatten')
            self.cnn_features = x

            with tf.variable_scope('fc_block'):
                units = [128, 8]
                for i in range(len(units)):
                    x = tf.layers.dense(inputs=x,
                                        units=units[i],
                                        activation=tf.nn.leaky_relu,
                                        trainable=trainable,
                                        name='fc_{}'.format(i))
                logits = tf.layers.dense(inputs=x,
                                         units=4,
                                         activation=None,
                                         trainable=trainable,
                                         name='fc_2')

            output = tf.nn.softmax(logits * self.config.meta.logits_scale)
            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return output, param

    def build_critic_net(self, scope):
        #x = self.cnn_features
        x = self.state
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_block'):
                filters = [16, 32, 64, 64]
                kernel_sizes = [1, 1, 3, 4]
                strides = [1, 1, 2, 2]
                for i in range(len(filters)):
                    x = tf.layers.conv2d(inputs=x,
                                         filters=filters[i],
                                         kernel_size=kernel_sizes[i],
                                         strides=(strides[i], strides[i]),
                                         padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         name='conv2d_{}'.format(i))
            x = tf.layers.flatten(inputs=x, name='flatten')

            with tf.variable_scope('fc_block'):
                units = [64, 8]
                for i in range(len(units)):
                    x = tf.layers.dense(inputs=x,
                                        units=units[i],
                                        activation=tf.nn.leaky_relu,
                                        name='fc_{}'.format(i))
                self.check_value = x
                value = tf.layers.dense(inputs=x,
                                        units=1,
                                        activation=None,
                                        name='fc_2')
                value = tf.squeeze(value)
            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return value, param

    def get_value(self, state):
        feed_dict = {self.state: state}
        value = self.sess.run(self.value, feed_dict)
        return value

