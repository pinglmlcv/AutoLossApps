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

from models.controllers import MlpPPO
from models import controller
from models import two_rooms
from models import three_rooms
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
            self.controller = controller.Controller(
                config,
                self.sess,
                exp_name=exp_name+'/controller'
            )
        elif config.meta.controller == 'MlpPPO':
            self.controller = MlpPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        else:
            raise Exception('Invalid controller name')

        self.env_list = []
        # First task: find the door of the first room
        goal_1 = [(4, 6)]
        init_1 = []
        for i in range(1,8,1):
            for j in range(1,6,1):
                init_1.append((i,j))
        self.env_list.append(three_rooms.Env3Rooms(config,
                                                   optional_goals=goal_1,
                                                   optional_inits=init_1))

        # Second task: find the door of the second room
        goal_2 = [(4, 12)]
        init_2 = [(4, 7)]
        self.env_list.append(three_rooms.Env3Rooms(config,
                                                   optional_goals=goal_2,
                                                   optional_inits=init_2))

        # Third task: find the target in the last room
        goal_3 = []
        for i in range(1,8,1):
            for j in range(14,20,1):
                goal_3.append((i,j))
        init_3 = [(4, 13)]
        self.env_list.append(three_rooms.Env3Rooms(config,
                                                   optional_goals=goal_3,
                                                   optional_inits=init_3))

        # Target_task: find the path from the first room to the target
        goal_4 = goal_3
        init_4 = init_1
        self.env_list.append(three_rooms.Env3Rooms(config,
                                                   optional_goals=goal_4,
                                                   optional_inits=init_4))

        self.agent = \
            gridworld_agent.AgentGridWorld(
                config,
                self.sess,
                exp_name='{}/agent'.format(exp_name),
            )


    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        agent = self.agent
        env_list = self.env_list

        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBufferMeta_on_policy = replaybuffer.ReplayBuffer(
            config.meta.buffer_size, keys=keys)
        replayBufferMeta_off_policy = replaybuffer.ReplayBuffer(
            config.meta.buffer_size, keys=keys)
        meta_training_history = deque(maxlen=10)
        nEnv = len(env_list)
        # ----The epsilon decay schedule.----
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)

        # ----Initialize controller and agent.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)
        # ----Initialize agent.----
        agent.initialize_weights()

        # ----Initialize performance matrix.----
        performance_matrix_init = np.zeros((nEnv, config.agent.total_episodes + 1))
        for i in range(nEnv):
            performance_matrix_init[i, 0] = \
                self.test_agent(agent, env_list[i])

        for ep_meta in range(config.meta.total_episodes):
            logger.info('######## meta_episodes {} #########'.format(ep_meta))
            start_time = time.time()
            replayBufferAgent_list = []
            for i in range(nEnv):
                replayBufferAgent_list.append(
                    replaybuffer.ReplayBuffer(config.agent.buffer_size))

            # ----Initialize agent.----
            agent.initialize_weights()
            agent.update_steps = 0

            # ----Initialize performance matrix and lesson probability.----
            performance_matrix  = performance_matrix_init.copy()
            lesson_history = []
            lesson_prob = self.calc_lesson_prob(lesson_history)
            meta_transitions = []

            # ----Start an episode.----

            # NOTE: We call M timesteps as an episode. Because the length of an
            # episode is unfixed so the training steps of each lesson is
            # different if we use a true episode as a training unit.
            meta_state = self.get_meta_state(performance_matrix[:, 0])
            for ep in range(config.agent.total_episodes):
                # TODO: choose with probability
                if config.meta.off_policy:
                    meta_action, pi = self.controller_behavior([meta_state], ep, 0)
                else:
                    meta_action, pi = controller.run_step([meta_state], ep, 0)
                value = controller.get_value([meta_state])
                #logger.info(value)
                logger.info('pi: {}'.format(pi))
                logger.info('action: {}'.format(meta_action))
                lesson = meta_action
                if not config.agent.mute:
                    logger.info('=================')
                    logger.info('episodes: {}, lesson: {}'.format(ep, lesson))

                if lesson < nEnv - 1:
                    # ----Training on a subtask and fine-tuning on the target
                    # task.----
                    # Training on subtask
                    student = lesson
                    epsilon = epsilons[min(agent.update_steps,
                                           config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_lesson(agent,
                                                env_list[student],
                                                replayBufferAgent_list[student],
                                                epsilon,
                                                mute=config.agent.mute)

                    # Training on target task
                    student = nEnv - 1
                    epsilon = epsilons[min(agent.update_steps,
                                           config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_lesson(agent,
                                                env_list[student],
                                                replayBufferAgent_list[student],
                                                epsilon,
                                                mute=config.agent.mute)

                elif lesson == nEnv - 1:
                    # ----Only training on target task.----
                    student = nEnv - 1
                    epsilon = epsilons[min(agent.update_steps,
                                           config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_lesson(agent,
                                                env_list[student],
                                                replayBufferAgent_list[student],
                                                epsilon,
                                                mute=config.agent.mute)
                else:
                    logger.error('Wrong action')

                # ----Update performance matrix.----
                self.update_performance_matrix(performance_matrix, ep)
                meta_reward = self.get_meta_reward_real_time(
                    performance_matrix, ep)

                # ----Update lesson probability.----
                lesson_history.append(lesson)
                lesson_prob = self.calc_lesson_prob(lesson_history)

                # ----Print training process.----
                if ep % config.agent.valid_frequency == 0:
                    logger.info('ep: {}, lesson_prob: {}'\
                                .format(ep, lesson_prob))
                    logger.info('pm of agent: {}'\
                                .format(performance_matrix[:, ep+1]))

                # ----Save transition.----
                meta_state_new = self.get_meta_state(performance_matrix[:, ep+1])
                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': meta_reward,
                              'target_value': meta_reward,
                              'next_state': meta_state_new}
                meta_transitions.append(transition)
                meta_state = meta_state_new

                # ----Check terminate.----
                if self.check_terminate(performance_matrix):
                    break

                # ----End of an agent episode.----

            reward_decay(meta_transitions, config.meta.gamma)
            meta_reward_final = self.get_meta_reward_final(performance_matrix)
            for t in meta_transitions:
                t['reward'] = meta_reward_final + t['reward']
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
            action = agent.run_step(state, epsilon=epsilon, greedy=config.agent.greedy)
            observation, reward, alive = env.update(action)
            next_state = preprocess(observation)
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state}
            replayBuffer.add(transition)
            batch_size = min(replayBuffer.population, config.agent.batch_size)
            batch = replayBuffer.get_batch(batch_size)
            # NOTE: q value of hybrid agent is used to regularize task
            # agent
            q_expert = self.agent.calc_q_value(batch['state'])
            batch['q_expert'] = q_expert
            agent.update(batch, mode='train')
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
        for step in range(config.agent.lesson_length_distill):
            batch_size = min(replayBuffer.population, config.agent.batch_size)
            batch = replayBuffer.get_batch(batch_size)
            q_expert = agent_t.calc_q_value(batch['state'])
            batch['q_expert'] = q_expert
            agent_s.update(batch)
            if agent_s.update_steps % config.agent.synchronize_frequency == 0:
                agent_s.sync_net()

    def distill_agent_one_step(self, agent_t, agent_s, env, replayBuffer,
                               epsilon, mute=False):
        config = self.config
        batch_size = min(replayBuffer.population, config.agent.batch_size)
        batch = replayBuffer.get_batch(batch_size)
        q_expert = agent_t.calc_q_value(batch['state'])
        batch['q_expert'] = q_expert
        agent_s.update(batch)
        if agent_s.update_steps % config.agent.synchronize_frequency == 0:
            agent_s.sync_net()

    def test(self, load_model, ckpt_num=None):
        config = self.config
        agent = self.agent
        env = self.env_list[-1]

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
            #epsilon = 0.05
            epsilon = 0.0
            total_reward = 0
            for i in range(config.agent.total_steps_test):
                action = agent.run_step(state, greedy=config.agent.greedy,
                                        epsilon=epsilon)
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

    def update_performance_matrix(self, performance_matrix, ep):
        # ----Update performance matrix.----
        #   Using expontional moving average method to update the entries of
        #   performance matrix masked by `mask`, while the other entries
        #   remains unchanged. Each entry represent an agent-task pair
        nEnv, _ = performance_matrix.shape
        ema_decay = config.meta.ema_decay_state
        for i in range(nEnv):
            r = self.test_agent(self.agent,
                                self.env_list[i],
                                mute=True)
            performance_matrix[i, ep + 1] =\
                performance_matrix[i, ep] * ema_decay\
                + r * (1 - ema_decay)

    def calc_lesson_prob(self, lesson_history):
        # ----Update lesson probability.----
        # Update the probability of each lesson in the lastest 50 episodes
        win_size = 20
        nLesson = len(self.env_list)
        s = lesson_history[max(0, len(lesson_history) - win_size) :]
        lesson_prob = np.zeros(nLesson)
        for l in s:
            lesson_prob[l] += 1
        return lesson_prob / max(1, len(s))

    def get_meta_state(self, pm):
        return pm

    def get_meta_reward_final(self, performance_matrix):
        return 0

    def get_meta_reward_real_time(self, matrix, ep):
        reward = matrix[-1, ep + 1] - matrix[-1, ep]
        return reward

    def baseline_multi(self):
        config = self.config
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)
        agent = self.agent
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

    def check_terminate(self, performance_matrix):
        return False


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
        trainer.train_meta(save_model=True)
        #trainer.train_meta(save_model=True, load_model=controller_ckpt)
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


