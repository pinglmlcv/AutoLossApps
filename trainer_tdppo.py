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

def reward_decay(meta_transitions, gamma):
    for i in range(len(meta_transitions) - 2, -1, -1):
        meta_transitions[i]['reward'] += meta_transitions[i+1]['reward'] * gamma


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
            self.controller = ppo_controller.MlpPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        elif config.meta.controller == 'CNNPPO':
            self.controller = tdppo_controller.CNNPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        else:
            raise Exception('Invalid controller name')

        self.env = two_rooms.Env2Rooms(config, default_goal=(3, 3))

    def train(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        keys = ['state', 'action', 'reward', 'target_value', 'next_state']
        replayBuffer = replaybuffer.ReplayBuffer(config.agent.buffer_size,
                                                 keys=keys)

        # ----Initialize performance matrix.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)

        for ep in range(config.agent.total_episodes):
            logger.info('######## episodes {} #########'.format(ep))
            start_time = time.time()
            self.train_agent_one_lesson(controller,
                                        self.env,
                                        replayBuffer,
                                        epsilon,
                                        mute=config.agent.mute)

        self.test_agent(self.controller,
                        self.env,
                        num_episodes=100,
                        mute=False)

        # ----End of a meta episode.----
        logger.info('running time: {}'.format(time.time() - start_time))
        logger.info('#################')

    def train_agent_one_lesson(self, agent, env, replayBuffer, epsilon,
                               mute=False):
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
                                        num_episodes=1,
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
        config = self.config
        dim_a = config.meta.dim_a
        lp = (lp - 0.5) * 2
        return np.concatenate((pm.flatten(), lp))

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
        return reward


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
        trainer.train(save_model=False)
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


