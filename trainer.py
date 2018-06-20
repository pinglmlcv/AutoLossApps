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

        # ----three agents----
        self.controller = controller.Controller(config, self.sess,
                                               exp_name=exp_name+'/controller')
        self.env_list = []
        self.agent_list = []
        default_goals = [(2, 1),
                         (2, 5),
                         (6, 1),
                         (6, 5),
                         (2, 14),
                         (2, 18),
                         (6, 14),
                         (6, 18),
                         (4, 10)]
        for goal in default_goals:
            self.env_list.append(two_rooms.Env2Rooms(config, default_goal=goal))
        self.env_list.append(two_rooms.Env2Rooms(config,
                                                 default_goals=default_goals))

        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent0'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent1'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent2'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent3'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent4'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent5'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent6'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent7'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent8'))
        self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                                self.sess,
                                                exp_name=exp_name+'/agent9'))

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
        for step in range(config.meta.total_steps_distill):
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

    def train(self, load_model=None, save_model=None):
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

            if lesson < 10:
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
                teacher = lesson - 10
                student = 9
                epsilon = epsilons[min(ep_lesson[teacher] + ep_lesson[lesson],
                                       config.agent.epsilon_decay_steps-1)]
                self.distill_agent_one_step(self.agent_list[teacher],
                                            self.agent_list[student],
                                            self.env_list[teacher],
                                            replayBufferAgent_list[teacher],
                                            epsilon)
                #ep_lesson[lesson] += 1
                #if ep_lesson[lesson] % config.agent.valid_frequency == 0:
                #    logger.info('++++test agent {}++++'.format(student))
                #    self.test_agent(self.agent_list[student],
                #                    self.env_list[student])
                #    logger.info('++++++++++')
            logger.info('=================')

    def test(self, load_model, ckpt_num=None):
        config = self.config

        self.agent.initialize_weights()
        self.agent.load_model(load_model)
        self.test_agent(self.agent, self.env)


    def test_agent(self, agent, env):
        config = self.config

        total_reward_aver = 0
        success_rate = 0
        for ep in range(config.agent.total_episodes_test):
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
        success_rate /= config.agent.total_episodes_test
        total_reward_aver /= config.agent.total_episodes_test
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
            '{}/agent1'.format(argv[1])
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


