""" The module for training autoLoss """
# __Author__ == "Haowen Xu"
# __Date__ == "06-18-2018"

import tensorflow as tf
import numpy as np
import logging
import os
import sys
import math
import socket
from time import gmtime, strftime
import time
import random
from collections import deque

from models import tdppo_controller
from models import controller
from models import two_rooms
from models import gridworld_agent
import utils
from utils import replaybuffer
from utils.data_process import preprocess

root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def area_under_curve(curve, strategy='square'):
    if len(curve) == 0:
        return 0
    else:
        if strategy == 'square':
            weights = np.square(np.linspace(0, 1, len(curve)))
            return sum(np.array(curve) * weights) / len(curve)
        elif strategy == 'linear':
            weights = np.linspace(0, 1, len(curve))
            return sum(np.array(curve) * weights) / len(curve)
        elif strategy == 'uniform':
            return sum(np.array(curve)) / len(curve)

def reward_decay(transitions, gamma):
    for i in range(len(transitions) - 2, -1, -1):
        transitions[i]['target_value'] += transitions[i+1]['target_value'] * gamma


class MlpPPO(tdppo_controller.BasePPO):
    def __init__(self, config, sess, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

    def _build_placeholder(self):
        config = self.config.meta
        dim_s = config.dim_s
        with tf.variable_scope('placeholder'):
            self.state = tf.placeholder(tf.float32,
                                        shape=[None, dim_s],
                                        name='state')
            self.action = tf.placeholder(tf.int32, shape=[None],
                                         name='action')
            self.reward = tf.placeholder(tf.float32, shape=[None],
                                         name='reward')
            self.target_value = tf.placeholder(tf.float32,
                                               shape=[None],
                                               name='target_value')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dim_h = self.config.meta.dim_h
            dim_a = self.config.meta.dim_a
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

            output = tf.nn.softmax(logits * self.config.meta.logits_scale)

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

    def run_step(self, states, ep, epsilon=0):
        dim_a = self.config.meta.dim_a
        #return ep % 5, ep % 5
        if random.random() < epsilon:
            action = random.randint(0, self.config.meta.dim_a - 1)
            return action, 'random'
        else:
            pi = self.sess.run(self.pi, {self.state: states})[0]
            action = np.random.choice(dim_a, 1, p=pi)[0]
            return action, pi

    def get_value(self, state):
        feed_dict = {self.state: state}
        value = self.sess.run(self.value, feed_dict)
        return value


class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, exp_name=None):
        self.config = config

        hostname = socket.gethostname()
        hostname = '-'.join(hostname.split('.')[0:2])
        datetime = strftime('%m-%d-%H-%M', gmtime())
        if not exp_name:
            exp_name = '{}_{}'.format(hostname, datetime)
        logger.info('exp_name: {}'.format(exp_name))

        gpu_options = tf.GPUOptions(allow_growth=True)
        configProto = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=configProto)
        self.auc_baseline = None

        if config.meta.controller == 'designed':
            self.controller = controller.Controller(config, self.sess,
                exp_name=exp_name+'/controller')
        elif config.meta.controller == 'MlpPPO':
            self.controller = MlpPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        else:
            raise Exception('Invalid controller name')

        self.env_list = []
        self.agent_list = []
        optional_goals = [[(2, 1), (2, 5)],
                          [(6, 1), (6, 5)],
                          [(2, 15), (2, 17)],
                          [(6, 15), (6, 17)],
                         ]
        #optional_goals = [(2, 1),
        #                  (2, 5),
        #                  (6, 1),
        #                  (6, 5),
        #                  (2, 14),
        #                  (2, 18),
        #                  (6, 14),
        #                  (6, 18),
        #                  (4, 10)]
        #optional_goals = [(3, 3),
        #                  (5, 17)]
        #optional_goals = [(4, 10)]
        #for goal in optional_goals:
        #    self.env_list.append(two_rooms.Env2Rooms(config, default_goal=goal))
        for goal in optional_goals:
            self.env_list.append(two_rooms.Env2Rooms(config, optional_goals=goal))
        #self.env_list.append(two_rooms.Env2Rooms(config,
        #                                         optional_goals=optional_goals))
        #optional_goals = [(2, 2),
        #                  (2, 3),
        #                  (2, 4),
        #                  (3, 2),
        #                  (3, 3),
        #                  (3, 4),
        #                  (4, 2),
        #                  (4, 3),
        #                  (4, 4)]
        optional_goals = [(2, 1),
                          (2, 5),
                          (6, 1),
                          (6, 5),
                          (2, 15),
                          (2, 17),
                          (6, 15),
                          (6, 17),
                         ]
        self.env_list.append(two_rooms.Env2Rooms(config,
                                                 optional_goals=optional_goals))

        for i in range(len(self.env_list) - 1):
            self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                   self.sess,
                                   exp_name='{}/agent{}'.format(exp_name, i)))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                               self.sess,
                               exp_name='{}/agent_hybrid'.format(exp_name),
                               hybrid=True))

        self.target_agent_id = len(self.agent_list) - 1

    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBufferMeta_on_policy = replaybuffer.ReplayBuffer(
            config.meta.buffer_size, keys=keys)
        replayBufferMeta_off_policy = replaybuffer.ReplayBuffer(
            config.meta.buffer_size, keys=keys)
        meta_training_history = deque(maxlen=10)
        # ----The epsilon decay schedule.----
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)
        nAgent = self.target_agent_id + 1

        # ----Initialize performance matrix.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)
        for agent in self.agent_list:
            agent.initialize_weights()
        performance_matrix_init = np.zeros((nAgent, nAgent,
                                            config.agent.total_episodes + 1))
        for i in range(nAgent):
            performance_matrix_init[i, i, 0] = \
                self.test_agent(self.agent_list[i], self.env_list[i])

        for ep_meta in range(config.meta.total_episodes):
            logger.info('######## meta_episodes {} #########'.format(ep_meta))
            start_time = time.time()
            replayBufferAgent_list = []
            for i in range(len(self.agent_list)):
                replayBufferAgent_list.append(
                    replaybuffer.ReplayBuffer(config.agent.buffer_size))

            # ----Initialize agents.----
            for agent in self.agent_list:
                agent.initialize_weights()

            # ----Initialize performance matrix and lesson probability.----
            performance_matrix = performance_matrix_init.copy()
            lesson_history = []
            lesson_prob = self.calc_lesson_prob(lesson_history)
            meta_transitions = []
            target_agent_performance = []

            # ----Start a meta episode.----
            # curriculum content:
            #   0: train agent0
            #   1: train agent1
            #   2: distill from agent0 to agent1

            # NOTE: We call M timesteps as an episode. Because the length of an
            # episode is unfixed so the training steps of each lesson is
            # different if we use a true episode as a lesson.
            meta_state = self.get_meta_state(performance_matrix[:, :, 0],
                                             lesson_prob)
            for ep in range(config.agent.total_episodes):
                meta_action, pi = controller.run_step([meta_state], ep, 0)
                value = controller.get_value([meta_state])
                #logger.info(value)
                logger.info(pi)
                lesson = meta_action
                if not config.agent.mute:
                    logger.info('=================')
                    logger.info('episodes: {}, lesson: {}'.format(ep, lesson))

                #if lesson < nAgent:
                #    student = lesson
                #    epsilon = epsilons[min(self.agent_list[student].update_steps,
                #                           config.agent.epsilon_decay_steps-1)]
                #    self.train_agent_one_lesson(self.agent_list[student],
                #                                self.env_list[student],
                #                                replayBufferAgent_list[student],
                #                                epsilon,
                #                                mute=config.agent.mute)
                #else:
                #    # TODO: For brevity, we use the expert net to sample actions.
                #    # Previous study showed that sampling actions from student
                #    # net gives a better result, we might try it later.
                #    teacher = lesson - nAgent
                #    student = nAgent - 1
                #    epsilon = epsilons[min(self.agent_list[teacher].update_steps,
                #                           config.agent.epsilon_decay_steps-1)]
                #    self.distill_agent_one_lesson(self.agent_list[teacher],
                #                                  self.agent_list[student],
                #                                  self.env_list[teacher],
                #                                  replayBufferAgent_list[teacher],
                #                                  epsilon,
                #                                  mute=config.agent.mute)

                if lesson < nAgent - 1:
                    student = lesson
                    epsilon = epsilons[min(self.agent_list[student].update_steps,
                                           config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_lesson(self.agent_list[student],
                                                self.env_list[student],
                                                replayBufferAgent_list[student],
                                                epsilon,
                                                mute=config.agent.mute)
                else:
                    # TODO: For brevity, we use the expert net to sample actions.
                    # Previous study showed that sampling actions from student
                    # net gives a better result, we might try it later.
                    student = nAgent - 1
                    for steps in range(int(config.agent.lesson_length / nAgent)):
                        for i in range(nAgent - 1):
                            teacher = i
                            self.distill_agent_one_step(
                                self.agent_list[teacher],
                                self.agent_list[student],
                                self.env_list[teacher],
                                replayBufferAgent_list[teacher],
                                0,
                                mute=config.agent.mute)

                # ----Update performance matrix.----
                mask = np.zeros((nAgent, nAgent), dtype=int)
                mask[student, student] = 1
                self.update_performance_matrix(performance_matrix, ep, mask)
                meta_reward = self.get_meta_reward_real_time(
                    performance_matrix, ep)

                #logger.info('performance_matrix: {}'.format(performance_matrix[:, :, ep+1]))

                #if self.agent_list[student].update_steps % config.agent.valid_frequency == 0:
                #    logger.info('++++test agent {}++++'.format(student))
                #    logger.info('update times:
                #    {}'.format(self.agent_list[student].update_steps))
                #    rew = []
                #    for i in range(1):
                #        r = self.test_agent(self.agent_list[student],
                #                            self.env_list[student],
                #                            mute=False)
                #        rew.append(r)
                #    logger.info('++++++++')

                # ----Update lesson probability.----
                lesson_history.append(lesson)
                lesson_prob = self.calc_lesson_prob(lesson_history)
                if ep % config.agent.valid_frequency == 0:
                    logger.info('ep: {}, lesson_prob: {}'\
                                .format(ep, lesson_prob))
                    for i in range(nAgent):
                        logger.info('pm of agent{}: {}'\
                                    .format(i, performance_matrix[i, i, ep+1]))

                meta_state_new = self.get_meta_state(performance_matrix[:, :, ep+1],
                                                     lesson_prob)

                # ----Save transition.----
                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': meta_reward,
                              'target_value': meta_reward,
                              'next_state': meta_state_new}
                meta_transitions.append(transition)
                meta_state = meta_state_new

                # ----End of an agent episode.----

            for i in range(nAgent):
                total_reward_aver = self.test_agent(self.agent_list[i],
                                self.env_list[i],
                                mute=False)

            #curve = performance_matrix[nAgent - 1, nAgent - 1, :] + \
            #    performance_matrix[0, 0, :]
            #auc = area_under_curve(curve, self.config.meta.reward_strategy)
            #meta_training_history.append(auc)
            #mean = np.mean(meta_training_history)
            #logger.info('mean_performance: {}'.format(mean))
            meta_training_history.append(total_reward_aver)
            mean = np.mean(meta_training_history)
            logger.info('mean_total_reward: {}'.format(mean))

            reward_decay(meta_transitions, config.meta.gamma)
            for t in meta_transitions:
                replayBufferMeta_on_policy.add(t)
                replayBufferMeta_off_policy.add(t)

            # ----Update controller using PPO.----
            lr = config.meta.lr
            if controller.update_steps == 200:
                lr /= 10

            if ep_meta < config.meta.warmup_steps or not config.meta.one_step_td:
                # using on_policy batch if:
                # 1) using Monte Carlo method
                # 2) in warmup stage of one-step td method
                batch_size = min(config.meta.batch_size,
                                 replayBufferMeta_on_policy.population)
                for i in range(10):
                    batch = replayBufferMeta_on_policy.get_batch(batch_size)
                    controller.update_actor(batch, lr)
                    controller.update_critic(batch, lr)
            else:
                # using off-policy batch if:
                # 1) after warmup stage of one-step td method
                batch_size = min(config.meta.batch_size,
                                 replayBufferMeta_off_policy.population)
                for i in range(100):
                    batch = replayBufferMeta_off_policy.get_batch(batch_size)
                    next_value = controller.get_value(batch['next_state'])
                    gamma = config.meta.gamma
                    batch['target_value'] = batch['reward'] + gamma * next_value
                    controller.update_actor(batch, lr)
                    controller.update_critic(batch, lr)

            controller.sync_net()
            replayBufferMeta_on_policy.clear()

            # ----Save contrller.----
            if save_model and ep_meta % config.meta.save_frequency == 0:
                controller.save_model(ep_meta)

            # ----End of a meta episode.----
            logger.info('running time: {}'.format(time.time() - start_time))
            logger.info('#################')

    def train_agent_one_lesson(self, agent, env, replayBuffer, epsilon,
                               mute=False):
        # NOTE: We use M timesteps instead of an episode. Because the length of
        # an episode is unfixed so the training steps of each lesson is
        # different if we use a true episode as a lesson.
        config = self.config

        # ----Lesson version.----
        if env.current_state is None:
            env.reset_game_art()
            env.set_goal_position()
            env.set_init_position()
            observation, _, _ = \
                env.init_episode(config.emulator.display_flag)
            state = preprocess(observation)
        else:
            state = env.current_state

        for step in range(config.agent.lesson_length):
            action = agent.run_step(state, epsilon)
            observation, reward, alive = env.update(action)
            next_state = preprocess(observation)
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state}
            replayBuffer.add(transition)
            if replayBuffer.population > config.agent.batch_size:
                batch = replayBuffer.get_batch(config.agent.batch_size)
                # NOTE: q value of hybrid agent is used to regularize task
                # agent
                q_expert = self.agent_list[-1].calc_q_value(batch['state'])
                batch['q_expert'] = q_expert
                agent.update(batch)
                if agent.update_steps % config.agent.synchronize_frequency == 0:
                    agent.sync_net()

            state = next_state
            if not alive:
                # ----One episode finished, start another.----
                env.reset_game_art()
                env.set_goal_position()
                env.set_init_position()
                observation, _, _ = \
                    env.init_episode(config.emulator.display_flag)
                state = preprocess(observation)

        env.current_state = state

    def distill_agent_one_lesson(self, agent_t, agent_s, env, replayBuffer,
                                 epsilon, mute=False):
        # NOTE: We call M timesteps as an episode. Because the length of an
        # episode is unfixed so the training steps of each lesson is
        # different if we use a true episode as a lesson.

        config = self.config

        # ----do not sample new transitions----
        for step in range(config.agent.lesson_length):
            if replayBuffer.population > config.agent.batch_size:
                batch = replayBuffer.get_batch(config.agent.batch_size)
                q_expert = agent_t.calc_q_value(batch['state'])
                batch['q_expert'] = q_expert
                agent_s.update_distill(batch)
                if agent_s.update_steps % config.agent.synchronize_frequency == 0:
                    agent_s.sync_net()

        #if env.current_state is None:
        #    env.reset_game_art()
        #    env.set_goal_position()
        #    env.set_init_position()
        #    observation, _, _ = \
        #        env.init_episode(config.emulator.display_flag)
        #    state = preprocess(observation)
        #else:
        #    state = env.current_state

        #for step in range(config.agent.lesson_length):
        #    action = agent_t.run_step(state, epsilon)
        #    observation, reward, alive = env.update(action)
        #    next_state = preprocess(observation)
        #    transition = {'state': state,
        #                  'action': action,
        #                  'reward': reward,
        #                  'next_state': next_state}
        #    replayBuffer.add(transition)

        #    if replayBuffer.population > config.agent.batch_size:
        #        batch = replayBuffer.get_batch(config.agent.batch_size)
        #        q_expert = agent_t.calc_q_value(batch['state'])
        #        batch['q_expert'] = q_expert
        #        agent_s.update_distill(batch)
        #        if agent_s.update_steps % config.agent.synchronize_frequency == 0:
        #            agent_s.sync_net()

        #    state = next_state
        #    if not alive:
        #        # ----One episode finished, start another.----
        #        env.reset_game_art()
        #        env.set_goal_position()
        #        env.set_init_position()
        #        obervation, _, _ = \
        #            env.init_episode(config.emulator.display_flag)
        #        state = preprocess(observation)
        #env.current_state = state

    def distill_agent_one_step(self, agent_t, agent_s, env, replayBuffer,
                               epsilon, mute=False):
        config = self.config
        if replayBuffer.population > config.agent.batch_size:
            batch = replayBuffer.get_batch(config.agent.batch_size)
            q_expert = agent_t.calc_q_value(batch['state'])
            batch['q_expert'] = q_expert
            agent_s.update(batch)
            if agent_s.update_steps % config.agent.synchronize_frequency == 0:
                agent_s.sync_net()

    def test(self, load_model, ckpt_num=None):
        config = self.config
        agent = self.agent_list[self.target_agent_id]
        env = self.env_list[self.target_agent_id]

        agent.initialize_weights()
        agent.load_model(load_model)
        self.test_agent(agent, env)


    def test_agent(self, agent, env, num_episodes=None, mute=False):
        config = self.config
        if not num_episodes:
            num_episodes = config.agent.total_episodes_test

        total_reward_aver = 0
        success_rate = 0
        for ep in range(num_episodes):
            env.reset_game_art()
            env.set_goal_position()
            env.set_init_position()
            observation, reward, _ =\
                env.init_episode(config.emulator.display_flag)
            state = preprocess(observation)
            epsilon = 0
            total_reward = 0
            for i in range(config.agent.total_steps):
                action = agent.run_step(state, epsilon)
                observation, reward, alive = env.update(action)
                total_reward += reward
                next_state = preprocess(observation)
                state = next_state
                # ----Reach the goal or knock into the wall.----
                if not alive:
                    if reward == 1:
                        success_rate += 1
                    break
            total_reward_aver += total_reward
        success_rate /= num_episodes
        total_reward_aver /= num_episodes
        if not mute:
            logger.info('total_reward_aver: {}'.format(total_reward_aver))
            logger.info('success_rate: {}'.format(success_rate))

        # NOTE: set env.current_state to None because testing process
        # interrupts the training process
        env.current_state = None

        return total_reward_aver


    def update_performance_matrix(self, performance_matrix, ep, mask):
        # ----Update performance matrix.----
        #   Using expontional moving average method to update the entries of
        #   performance matrix masked by `mask`, while the other entries
        #   remains unchanged. Each entry represent an agent-task pair
        nAgent, nEnv = mask.shape
        ema_decay = config.meta.ema_decay_state
        for i in range(nAgent):
            for j in range(nEnv):
                if mask[i, j]:
                    r = self.test_agent(self.agent_list[i],
                                        self.env_list[j],
                                        num_episodes=50,
                                        mute=True)
                    performance_matrix[i, j, ep + 1] =\
                        performance_matrix[i, j, ep] * ema_decay\
                        + r * (1 - ema_decay)
                else:
                    performance_matrix[i, j, ep + 1] =\
                        performance_matrix[i, j, ep]

    def calc_lesson_prob(self, lesson_history):
        # ----Update lesson probability.----
        # Update the probability of each lesson in the lastest 50 episodes
        win_size = 20
        nLesson = 2 * self.target_agent_id + 1
        s = lesson_history[max(0, len(lesson_history) - win_size) :]
        lesson_prob = np.zeros(nLesson)
        for l in s:
            lesson_prob[l] += 1
        return lesson_prob / max(1, len(s))

    def get_meta_state(self, pm, lp):
        #lp = (lp - 0.5) * 2
        #return np.concatenate((pm.flatten(), lp))
        #return pm.flatten()
        a = np.diag(pm)
        mean = np.mean(a[:-1])
        diff = a - mean
        return np.append(a, diff)

    def get_meta_reward(self, curve):
        auc = area_under_curve(curve, self.config.meta.reward_strategy)
        logger.info('auc: {}'.format(auc))
        if not self.auc_baseline:
            meta_reward = 0
            self.auc_baseline = auc
        else:
            ema_decay = self.config.meta.ema_decay_auc_baseline
            meta_reward = auc - self.auc_baseline
            self.auc_baseline = self.auc_baseline * ema_decay\
                + auc * (1 - ema_decay)
        return meta_reward

    def get_meta_reward_real_time(self, matrix, ep):
        old = matrix[:, :, ep]
        new = matrix[:, :, ep + 1]
        reward = 0
        for i in range(len(self.agent_list)):
            reward += (new[i, i] - old[i, i])
        #reward = (new[-1,-1] - old[-1,-1])
        return reward

    def baseline_multi(self):
        config = self.config
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)
        agent = self.agent_list[-1]
        agent.hybrid = False
        agent.initialize_weights()
        env = self.env_list[-1]
        replayBufferAgent = replaybuffer.ReplayBuffer(config.agent.buffer_size)
        for ep in range(config.agent.total_episodes):
            epsilon = epsilons[min(agent.update_steps,
                                   config.agent.epsilon_decay_steps-1)]
            self.train_agent_one_lesson(agent,
                                        env,
                                        replayBufferAgent,
                                        epsilon,
                                        mute=config.agent.mute)
            if ep % 20 == 0:
                logger.info('ep: {}'.format(ep))
                self.test_agent(agent, env, num_episodes=20, mute=False)
        total_reward_aver = self.test_agent(agent, env, mute=False)


if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info(socket.gethostname())
    config_file = 'gridworld_tdppo.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)
    config.print_config()

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, exp_name=argv[1])

    if argv[2] == 'train':
        # ----Training----
        logger.info('TRAIN')
        controller_ckpt = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/{}/controller/'.format(argv[1])
        controller_ckpt = '/media/haowen/AutoLossApps/saved_models/test/controller/'
        #trainer.train_meta(save_model=True)
        trainer.train_meta(save_model=True, load_model=controller_ckpt)
    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        agent_dir = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/'\
            '{}/agent2'.format(argv[1])
        trainer.test(agent_dir)
    elif argv[2] == 'baseline_multi':
        # ----Baseline----
        logger.info('BASELINE')
        trainer.baseline_multi()
    elif argv[2] == 'generate':
        logger.info('GENERATE')
        trainer.generate(load_stud)


