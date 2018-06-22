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

from models import controller
from models import two_rooms
from models import gridworld_agent
import utils
from utils import replaybuffer
from utils.data_process import preprocess

root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

def discount_rewards(reward, final_reward):
    # TODO(haowen) Final reward + step reward
    reward_dis = np.array(reward) + np.array(final_reward)
    return reward_dis

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
        # ----single agent----
        #self.env = two_rooms.Env2Rooms(config, default_init=(3, 3))
        #self.agent = gridworld_agent.AgentGridWorld(config,
        #                                       self.sess,
        #                                       exp_name=exp_name+'/agent1')

        # ----multi agents----
        self.controller = controller.Controller(config, self.sess,
                                               exp_name=exp_name+'/controller')
        self.env_list = []
        self.agent_list = []
        #optional_goals = [(2, 1),
        #                  (2, 5),
        #                  (6, 1),
        #                  (6, 5),
        #                  (2, 14),
        #                  (2, 18),
        #                  (6, 14),
        #                  (6, 18),
        #                  (4, 10)]
        optional_goals = [(3, 3),
                          (5, 17)]
        self.target_agent_id = len(self.agent_list) - 1

        for goal in optional_goals:
            self.env_list.append(two_rooms.Env2Rooms(config, default_goal=goal))
        self.env_list.append(two_rooms.Env2Rooms(config,
                                                 optional_goals=optional_goals))

        for i in range(len(optional_goals) + 1):
            self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                   self.sess,
                                   exp_name='{}/agent{}'.format(exp_name, i)))

    def train_agent(self, save_model=None, load_model=None):
        config = self.config
        replayBuffer = replaybuffer.ReplayBuffer(config.agent.buffer_size)
        agent = self.agent
        env = self.env
        # The epsilon decay schedule
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)
        replayBuffer.clear()
        # ----Initialize agent.----
        self.agent.initialize_weights()
        if load_model:
            agent.load_model(load_model)

        total_reward_aver = 0
        for ep in range(config.agent.total_episodes):
            logger.info('=================')
            logger.info('episodes: {}'.format(ep))

            env.reset_game_art()
            env.set_goal_position()
            env.set_init_position()
            observation, reward, _ =\
                env.init_episode(config.emulator.display_flag)
            state = preprocess(observation)
            epsilon = epsilons[min(ep,
                                   config.agent.epsilon_decay_steps-1)]

            # ----Running one episode.----
            total_reward = 0
            for step in range(config.agent.total_steps):
                action = agent.run_step(state, epsilon)
                observation, reward, alive = env.update(action)
                total_reward += reward
                next_state = preprocess(observation)
                transition = {'state': state,
                              'action': action,
                              'reward': reward,
                              'next_state': next_state}
                replayBuffer.add(transition)

                if replayBuffer.population > config.agent.batch_size:
                    batch = replayBuffer.get_batch(config.agent.batch_size)
                    agent.update(batch)
                agent.update_steps += 1
                if agent.update_steps % config.agent.synchronize_frequency == 0:
                    agent.sync_net()
                state = next_state
                if not alive:
                    break
            if alive:
                print('failed')
            #print(total_reward)
            total_reward_aver += total_reward
            if ep % 1 == 0:
                print(total_reward_aver / 1)
                total_reward_aver = 0
            if (ep % 100 == 0) and save_model:
                agent.save_model(ep)

    def train_agent_one_ep(self, agent, env, replayBuffer, epsilon):
        config = self.config

        env.reset_game_art()
        env.set_goal_position()
        env.set_init_position()
        observation, reward, _ =\
            env.init_episode(config.emulator.display_flag)
        state = preprocess(observation)

        # ----Running one episode.----
        total_reward = 0
        for step in range(config.agent.total_steps):
            action = agent.run_step(state, epsilon)
            observation, reward, alive = env.update(action)
            total_reward += reward
            next_state = preprocess(observation)
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state}
            replayBuffer.add(transition)

            if replayBuffer.population > config.agent.batch_size:
                batch = replayBuffer.get_batch(config.agent.batch_size)
                agent.update(batch)
            agent.update_steps += 1
            if agent.update_steps % config.agent.synchronize_frequency == 0:
                agent.sync_net()
            state = next_state
            if not alive:
                break
        if alive:
            logger.info('failed')
        logger.info(total_reward)

    def distill_agent_one_ep(self, agent_t, agent_s, env, replayBuffer, epsilon):
        config = self.config

        env.reset_game_art()
        env.set_goal_position()
        env.set_init_position()
        observation, reward, _ =\
            env.init_episode(config.emulator.display_flag)
        state = preprocess(observation)

        total_reward = 0
        # ----Running one episode.----
        for step in range(config.agent.total_steps):
            action = agent_t.run_step(state, epsilon)
            observation, reward, alive = env.update(action)
            total_reward += reward
            next_state = preprocess(observation)
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'next_state': next_state}
            replayBuffer.add(transition)

            if replayBuffer.population > config.agent.batch_size:
                batch = replayBuffer.get_batch(config.agent.batch_size)
                q_expert = agent_t.calc_q_value(batch['state'])
                batch['q_expert'] = q_expert
                agent_s.update_distill(batch)
            agent_s.update_steps += 1
            if agent_s.update_steps % config.agent.synchronize_frequency == 0:
                agent_s.sync_net()
            state = next_state
            if not alive:
                break
        if alive:
            logger.info('failed')
        logger.info(total_reward)

    def distill_agent_one_step(self, agent_t, agent_s, env, replayBuffer, epsilon):
        config = self.config

        if replayBuffer.population > config.agent.batch_size:
            batch = replayBuffer.get_batch(config.agent.batch_size)
            q_expert = agent_t.calc_q_value(batch['state'])
            batch['q_expert'] = q_expert
            agent_s.update_distill(batch)
        agent_s.update_steps += 1
        if agent_s.update_steps % config.agent.synchronize_frequency == 0:
            agent_s.sync_net()

    def train(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        replayBufferAgent_list = []
        for i in range(len(self.agent_list)):
            replayBufferAgent_list.append(
                replaybuffer.ReplayBuffer(config.agent.buffer_size))

        # The epsilon decay schedule
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)

        # ----Initialize agents.----
        for agent in self.agent_list:
            agent.initialize_weights()
        ep_lesson = [0] * (len(self.agent_list) * 2 - 1)

        # curriculum content:
        #   0: train agent0
        #   1: train agent1
        #   2: train agent2
        #   3: distill from agent0 to agent2
        #   4: distill from agent1 to agent2
        for ep in range(config.agent.total_episodes):
            lesson = controller.run_step(0, ep)
            logger.info('=================')
            logger.info('episodes: {}, lesson: {}'.format(ep, lesson))

            if lesson <= self.target_agent_id:
                epsilon = epsilons[min(ep_lesson[lesson],
                                       config.agent.epsilon_decay_steps-1)]
                self.train_agent_one_ep(self.agent_list[lesson],
                                        self.env_list[lesson],
                                        replayBufferAgent_list[lesson],
                                        epsilon)
                ep_lesson[lesson] += 1
                if ep_lesson[lesson] % config.agent.valid_frequency == 0:
                    logger.info('++++test agent {}++++'.format(lesson))
                    self.test_agent(self.agent_list[lesson],
                                    self.env_list[lesson])
                    logger.info('++++++++++')
            else:
                # TODO: For brevity, we use the expert net to sample actions.
                # Previous study showed that sampling actions from student net
                # gives a better result, we might try it later.
                teacher = lesson - self.target_agent_id - 1
                student = self.target_agent_id
                epsilon = epsilons[min(ep_lesson[teacher] + ep_lesson[lesson],
                                       config.agent.epsilon_decay_steps-1)]
                self.distill_agent_one_ep(self.agent_list[teacher],
                                            self.agent_list[student],
                                            self.env_list[teacher],
                                            replayBufferAgent_list[teacher],
                                            epsilon)
                ep_lesson[lesson] += 1
                if ep_lesson[lesson] % config.agent.valid_frequency == 0:
                    logger.info('++++test agent {}++++'.format(student))
                    self.test_agent(self.agent_list[student],
                                    self.env_list[student])
                    logger.info('++++++++++')
            logger.info('=================')

            if save_model and ep % config.agent.save_frequency == 0:
                for agent in self.agent_list:
                    agent.save_model(ep)

    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        replayBufferMeta = replayBuffer(config.meta.buffer_size)
        # The epsilon decay schedule
        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)
        nAgent = self.target_agent_id + 1
        ema_decay = config.meta.ema_decay
        for ep_meta in range(config.meta.total_episodes):
            logger.info('#################')
            logger.info('meta_episodes: {}'.format(ep_meta))
            replayBufferAgent_list = []
            for i in range(len(self.agent_list)):
                replayBufferAgent_list.append(
                    replaybuffer.ReplayBuffer(config.agent.buffer_size))

            # ----Initialize agents.----
            for agent in self.agent_list:
                agent.initialize_weights()
            ep_lesson = [0] * (len(self.agent_list) * 2 - 1)

            # ----Initialize performance matrix.----
            performance_matrix = np.zeros((nAgent, nAgent,
                                           config.agent.total_episodes + 1))
            for i in range(nAgent):
                for j in range(nAgent):
                    performance_matrix[i, j, 0] = \
                        self.test_agent(self.agent_list[i], self.env_list[j])

            # ----Start a meta episode.----
            # curriculum content:
            #   0: train agent0
            #   1: train agent1
            #   2: train agent2
            #   3: distill from agent0 to agent2
            #   4: distill from agent1 to agent2
            for ep in range(config.agent.total_episodes):
                meta_state = self.get_meta_state(performance_matrix,
                                                 previous_action)
                meta_action = controller.run_step(meta_state, ep)
                lesson = meta_action
                logger.info('=================')
                logger.info('episodes: {}, lesson: {}'.format(ep, lesson))

                if lesson < nAgent:
                    student = lesson
                    epsilon = epsilons[min(ep_lesson[student],
                                        config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_ep(self.agent_list[student],
                                            self.env_list[student],
                                            replayBufferAgent_list[student],
                                            epsilon)
                    ep_lesson[student] += 1
                    #if ep_lesson[student] % config.agent.valid_frequency == 0:
                    #    logger.info('++++test agent {}++++'.format(student))
                    #    self.test_agent(self.agent_list[student],
                    #                    self.env_list[student])
                    #    logger.info('++++++++++')
                else:
                    # TODO: For brevity, we use the expert net to sample actions.
                    # Previous study showed that sampling actions from student
                    # net gives a better result, we might try it later.
                    teacher = lesson - nAgent
                    student = nAgent - 1
                    epsilon = epsilons[min(ep_lesson[teacher] + ep_lesson[student],
                                        config.agent.epsilon_decay_steps-1)]
                    self.distill_agent_one_ep(self.agent_list[teacher],
                                                self.agent_list[student],
                                                self.env_list[teacher],
                                                replayBufferAgent_list[teacher],
                                                epsilon)
                    ep_lesson[student] += 1
                    #if ep_lesson[student] % config.agent.valid_frequency == 0:
                    #    logger.info('++++test agent {}++++'.format(student))
                    #    self.test_agent(self.agent_list[student],
                    #                    self.env_list[student])
                    #    logger.info('++++++++++')
                # ----Update performance matrix.----
                mask = np.zeros((nAgent, nAgent), dtype=int)
                mask[student] = 1
                self.update_performance_matrix(performance_matrix, ep, mask)

                # ----Update lesson probability.----
                # TODO:
                self.update_lesson_probability()

                # ----Add transition to meta_buffer.----
                # TODO:

                # ----End of an agent episode.----
                logger.info('=================')

            # ----Calculate meta_reward.----
            # TODO:

            # ----Update controller using PPO.----
            # TODO:

            # ----End of a meta episode.----
            logger.info('#################')


    def test(self, load_model, ckpt_num=None):
        config = self.config
        agent = self.agent_list[self.target_agent_id]
        env = self.env_list[self.target_agent_id]

        agent.initialize_weights()
        agent.load_model(load_model)
        self.test_agent(agent, env)


    def test_agent(self, agent, env, max_episodes=None):
        config = self.config
        if not max_episodes:
            max_episodes = config.agent.total_episodes_test

        total_reward_aver = 0
        success_rate = 0
        for ep in range(max_episodes):
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
                if not alive:
                    success_rate += 1
                    break
            total_reward_aver += total_reward
        success_rate /= max_episodes
        total_reward_aver /= max_episodes
        logger.info('total_reward_aver: {}'.format(total_reward_aver))
        logger.info('success_rate: {}'.format(success_rate))
        return total_reward_aver


    def baseline(self):
        self.model_stud.initialize_weights()
        self.model_stud.train(save_model=True)
        if self.config.student_model_name == 'gan':
            self.model_stud.load_model(self.model_stud.task_dir)
            inps_baseline = self.model_stud.get_inception_score(500, splits=5)
            self.model_stud.generate_images(0)
            logger.info('inps_baseline: {}'.format(inps_baseline))
        return inps_baseline

    def generate(self, load_stud):
        self.model_stud.initialize_weights()
        self.model_stud.load_model(load_stud)
        self.model_stud.generate_images(0)

    def update_performance_matrix(self, performance_matrix, ep, mask):
        # ----Update performance matrix.----
        #   Using expontional moving average method to update the entries of
        #   performance matrix masked by `mask`, while the other entries
        #   remains unchanged. Each entry represent an agent-task pair
        nAgent, nEnv = mask.shape
        for i in range(nAgent):
            for j in range(nEnv):
                if mask[i, j]:
                    r = self.test_agent(self.agent_list[i],
                                        self.env_list[j],
                                        1)
                    performance_matrix[i, j, ep + 1] =\
                        performance_matrix[i, j, ep] * ema_decay\
                        + r * (1 - ema_decay)
                else:
                    performance_matrix[i, j, ep + 1] =\
                        performance_matrix[i, j, ep]

    def update_lesson_probability(self):
        pass


if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info(socket.gethostname())
    config_file = 'gridworld.cfg'
    config_path = os.path.join(root_path, 'config/' + config_file)
    config = utils.Parser(config_path)
    config.print_config()

    # ----Instantiate a trainer object.----
    trainer = Trainer(config, exp_name=argv[1])

    if argv[2] == 'train':
        # ----Training----
        logger.info('TRAIN')
        #trainer.train_agent(save_model=True)
        trainer.train(save_model=True)
    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        agent_dir = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/'\
            '{}/agent2'.format(argv[1])
        trainer.test(agent_dir)
    elif argv[2] == 'baseline':
        # ----Baseline----
        logger.info('BASELINE')
        baseline_accs = []
        for i in range(1):
            baseline_accs.append(trainer.baseline())
        logger.info(baseline_accs)
    elif argv[2] == 'generate':
        logger.info('GENERATE')
        trainer.generate(load_stud)


