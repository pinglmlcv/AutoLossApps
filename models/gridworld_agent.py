# __Author__ == "Haowen Xu"
# __Date__ == "06-15-2018"

import tensorflow as tf
import numpy as np
import math
import os
#from keras.layers import Dense
#from keras.optimizers import Adam
#from keras.models import Sequential
#from keras import backend as K

import utils
from models import layers
from models.basic_model import Basic_model
logger = utils.get_logger()

def dqn(x, name='dqn'):
    with tf.variable_scope(name):
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
                                     name='dqn_conv2d_{}'.format(i))
        x = tf.layers.flatten(inputs=x, name='flatten')
        out = []
        with tf.variable_scope('fc_block'):
            units = [128, 8, 4]
            for i in range(len(units)):
                x = tf.layers.dense(inputs=x,
                                    units=units[i],
                                    activation=tf.nn.leaky_relu,
                                    name='dqn_fc_{}'.format(i))
                out.append(x)
        return x, out


class AgentGridWorld(Basic_model):
    def __init__(self, config, sess, exp_name='agent_gridworld'):
        super(AgentGridWorld, self).__init__(config, sess, exp_name)
        self.update_steps = 0
        self.current_state = None
        #TODO: implement buffer
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def _build_placeholder(self):
        config = self.config
        dim_s_h = config.agent.dim_s_h
        dim_s_w = config.agent.dim_s_w
        dim_s_c = config.agent.dim_s_c
        dim_a = config.agent.dim_a
        with tf.variable_scope('placeholder'):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, dim_s_h, dim_s_w, dim_s_c],
                                        name='state')
            self.next_state = tf.placeholder(tf.float32,
                                        shape=[None, dim_s_h, dim_s_w, dim_s_c],
                                        name='next_state')
            self.reward = tf.placeholder(tf.float32,
                                         shape=[None,],
                                         name='reward')
            self.action = tf.placeholder(tf.int32,
                                         shape=[None,],
                                         name='action')
            self.q_expert = tf.placeholder(tf.float32,
                                           shape=[None, dim_a],
                                           name='q_expert')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _build_graph(self):
        config = self.config
        self.dqn, self.out = dqn(self.state, name='dqn')
        self.dqn_target, _ = dqn(self.next_state, name='dqn_target')
        b_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=self.exp_name+'/dqn')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=self.exp_name+'/dqn_target')

        q_target = self.reward + config.agent.gamma *\
            tf.reduce_max(self.dqn_target, axis=1)
        #NOTE: target network is fixed when we update behavior network
        q_target = tf.stop_gradient(q_target, name='q_target')

        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32),
                              self.action],
                             axis=1)
        self.q_wrt_a = tf.gather_nd(params=self.dqn,
                                    indices=a_indices, name='q')

        # dqn loss
        dqn_loss = tf.losses.mean_squared_error(labels=q_target,
                                                predictions=self.q_wrt_a,
                                                scope='dqn_loss')

        if config.meta.distill_mode == 'CE':
            distill_loss = tf.losses.softmax_cross_entropy(
                tf.nn.softmax(self.q_expert * config.meta.distill_temp),
                tf.nn.softmax(self.dqn * config.meta.distill_temp)
            )
        elif config.meta.distill_mode == 'KL':
            p_teacher = tf.nn.softmax(self.q_expert * config.meta.distill_temp)
            p_student = tf.nn.softmax(self.dqn * config.meta.distill_temp)
            distill_loss = tf.reduce_mean(tf.reduce_sum(
                p_teacher * tf.log(p_teacher / p_student)))


        self.train_op = tf.train.AdamOptimizer(self.lr)\
            .minimize(dqn_loss)

        self.distill_op = tf.train.AdamOptimizer(self.lr / 10)\
            .minimize(distill_loss)

        self.tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope=self.exp_name)
        self.saver = tf.train.Saver(var_list=self.tvars, max_to_keep=1)
        self.init = tf.variables_initializer(self.tvars)
        self.target_replace_op = [tf.assign(t, b)
                                  for t, b in zip(t_params, b_params)]
        self.q_target = q_target
        self.dqn_loss = dqn_loss
        self.distill_loss = distill_loss

    def sync_net(self):
        self.sess.run(self.target_replace_op)
        logger.info('target_network synchronized')

    def train(self, save_model=False):
        sess = self.sess
        config = self.config
        batch_size = config.batch_size
        dim_z = config.dim_z
        valid_frequency = config.valid_frequency_stud
        print_frequency = config.print_frequency_stud
        max_endurance = config.max_endurance_stud
        endurance = 0
        best_inps = 0
        inps_baseline = 0
        decay = config.metric_decay
        steps_per_iteration = config.disc_iters + config.gen_iters
        for step in range(config.max_training_step):
            if step % steps_per_iteration < config.disc_iters:
                # ----Update D network.----
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']

                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: True}
                sess.run(self.disc_train_op, feed_dict=feed_dict)
            else:
                # ----Update G network.----
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.is_training: True}
                sess.run(self.gen_train_op, feed_dict=feed_dict)

            if step % valid_frequency == 0:
                logger.info('========Step{}========'.format(step))
                logger.info(endurance)
                inception_score = self.get_inception_score(config.inps_batches)
                logger.info(inception_score)
                if inps_baseline > 0:
                    inps_baseline = inps_baseline * decay \
                        + inception_score[0] * (1 - decay)
                else:
                    inps_baseline = inception_score[0]
                logger.info('inps_baseline: {}'.format(inps_baseline))
                self.generate_images(step)
                endurance += 1
                if inps_baseline > best_inps:
                    best_inps = inps_baseline
                    endurance = 0
                    if save_model:
                        self.save_model(step)

            if step % print_frequency == 0:
                data = self.train_dataset.next_batch(batch_size)
                x = data['input']
                z = np.random.normal(size=[batch_size, dim_z]).astype(np.float32)
                feed_dict = {self.noise: z, self.real_data: x,
                             self.is_training: False}
                fetch = [self.gen_cost,
                         self.disc_cost_fake,
                         self.disc_cost_real]
                r = sess.run(fetch, feed_dict=feed_dict)
                logger.info('gen_cost: {}'.format(r[0]))
                logger.info('disc_cost fake: {}, real: {}'.format(r[1], r[2]))

            if endurance > max_endurance:
                break
        logger.info('best_inps: {}'.format(best_inps))

    def update(self, transition_batch):
        self.update_steps += 1
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        next_state = transition_batch['next_state']
        fetch = [self.train_op, self.dqn]
        feed_dict = {self.state: state,
                     self.next_state: next_state,
                     self.reward: reward,
                     self.action: action,
                     self.lr: self.config.agent.lr}
        _, pred = self.sess.run(fetch, feed_dict)

    def update_distill(self, transition_batch):
        self.update_steps += 1
        state = transition_batch['state']
        action = transition_batch['action']
        reward = transition_batch['reward']
        next_state = transition_batch['next_state']
        q_expert = transition_batch['q_expert']
        fetch = [self.distill_op, self.train_op]
        feed_dict = {self.state: state,
                     self.next_state: next_state,
                     self.reward: reward,
                     self.action: action,
                     self.q_expert: q_expert,
                     self.lr: self.config.agent.lr}
        self.sess.run(fetch, feed_dict)

    def run_step(self, state, epsilon=0):
        dim_a = self.config.agent.dim_a
        A = np.ones(dim_a, dtype=float) * epsilon / dim_a
        feed_dict = {self.state: [state]}
        q_values = self.sess.run(self.dqn, feed_dict)[0]
        #print(q_values)
        #print('q_values: {}'.format(q_values))
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        a = np.arange(dim_a)
        return np.random.choice(a, p=A)

    def calc_q_value(self, state):
        feed_dict = {self.state: state}
        q_value = self.sess.run(self.dqn, feed_dict)
        return q_value


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    gan = Gan(config)
