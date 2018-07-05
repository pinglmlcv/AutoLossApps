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

from models import ppo_controller
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
        # ----single agent----
        #self.env = two_rooms.Env2Rooms(config, default_init=(3, 3))
        #self.agent = gridworld_agent.AgentGridWorld(config,
        #                                       self.sess,
        #                                       exp_name=exp_name+'/agent1')

        # ----multi agents----
        if config.meta.controller == 'designed':
            self.controller = controller.Controller(config, self.sess,
                exp_name=exp_name+'/controller')
        elif config.meta.controller == 'MlpPPO':
            self.controller = ppo_controller.MlpPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        else:
            raise Exception('Invalid controller name')

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

        for goal in optional_goals:
            self.env_list.append(two_rooms.Env2Rooms(config, default_goal=goal))
        self.env_list.append(two_rooms.Env2Rooms(config,
                                                 optional_goals=optional_goals))

        for i in range(len(optional_goals) + 1):
            self.agent_list.append(gridworld_agent.AgentGridWorld(config,
                                   self.sess,
                                   exp_name='{}/agent{}'.format(exp_name, i)))
        self.target_agent_id = len(self.agent_list) - 1

    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        replayBufferMeta = replaybuffer.ReplayBuffer(config.meta.buffer_size)
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
            for j in range(nAgent):
                performance_matrix_init[i, j, 0] = \
                    self.test_agent(self.agent_list[i], self.env_list[j])

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

            # NOTE: `ep_agent` records how many episodes an agent has been
            # updated for
            ep_agent = [0] * len(self.agent_list)

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
            #   2: train agent2
            #   3: distill from agent0 to agent2
            #   4: distill from agent1 to agent2

            # NOTE: We call M timesteps as an episode. Because the length of an
            # episode is unfixed so the training steps of each lesson is
            # different if we use a true episode as a lesson.
            for ep in range(config.agent.total_episodes):
                meta_state = self.get_meta_state(performance_matrix[:, :, ep],
                                                 lesson_prob)
                meta_action, pi = controller.run_step(meta_state, ep)
                logger.info(pi)
                lesson = meta_action
                if not config.agent.mute:
                    logger.info('=================')
                    logger.info('episodes: {}, lesson: {}'.format(ep, lesson))

                if lesson < nAgent:
                    student = lesson
                    epsilon = epsilons[min(ep_agent[student],
                                           config.agent.epsilon_decay_steps-1)]
                    self.train_agent_one_lesson(self.agent_list[student],
                                                self.env_list[student],
                                                replayBufferAgent_list[student],
                                                epsilon,
                                                mute=config.agent.mute)
                    ep_agent[student] += 1
                else:
                    # TODO: For brevity, we use the expert net to sample actions.
                    # Previous study showed that sampling actions from student
                    # net gives a better result, we might try it later.
                    teacher = lesson - nAgent
                    student = nAgent - 1
                    epsilon = epsilons[min(ep_agent[teacher],
                                           config.agent.epsilon_decay_steps-1)]
                    self.distill_agent_one_lesson(self.agent_list[teacher],
                                                  self.agent_list[student],
                                                  self.env_list[teacher],
                                                  replayBufferAgent_list[teacher],
                                                  epsilon,
                                                  mute=config.agent.mute)
                    ep_agent[student] += 1

                # ----Update performance matrix.----
                mask = np.zeros((nAgent, nAgent), dtype=int)
                mask[student] = 1
                self.update_performance_matrix(performance_matrix, ep, mask)
                meta_reward = self.get_meta_reward_real_time(
                    performance_matrix, ep)

                #logger.info('performance_matrix: {}'.format(performance_matrix[:, :, ep+1]))

                #if ep_agent[student] % config.agent.valid_frequency == 0:
                #    logger.info('++++test agent {}++++'.format(student))
                #    logger.info('update times: {}'.format(ep_agent[student]))
                #    rew = []
                #    for i in range(1):
                #        r = self.test_agent(self.agent_list[student],
                #                            self.env_list[student],
                #                            mute=False)
                #        rew.append(r)
                #    logger.info('++++++++')


                # NOTE: Record performance of target agent whenever it has been
                # updated for `valid_frequency` steps. This recording is used
                # to evaluate the meta controller.
                # TODO: Both inefficient and inaccurate. Need to find a better
                # evaluation method.
                # ----option 1----
                # if student == nAgent - 1:
                #     if ep_agent[student] % config.agent.valid_frequency == 0:
                #         logger.info('++++test agent {}++++'.format(student))
                #         r = self.test_agent(self.agent_list[student],
                #                             self.env_list[student])
                #         target_agent_performance.append(r)
                #         logger.info('++++++++++')
                # ----option 2----
                # if student == nAgent - 1:
                #     target_agent_performance.append(
                #         performance_matrix[-1, -1, ep + 1])

                # ----Update lesson probability.----
                lesson_history.append(lesson)
                lesson_prob = self.calc_lesson_prob(lesson_history)
                if ep % 20 == 0:
                    logger.info('ep: {}, lesson_prob: {}'.format(ep, lesson_prob))

                # ----Save transition.----
                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': meta_reward,
                              'next_state': 0}
                meta_transitions.append(transition)

                # ----End of an agent episode.----

            self.test_agent(self.agent_list[0],
                            self.env_list[0],
                            num_episodes=100,
                            mute=False)
            self.test_agent(self.agent_list[1],
                            self.env_list[1],
                            num_episodes=100,
                            mute=False)
            self.test_agent(self.agent_list[2],
                            self.env_list[2],
                            num_episodes=100,
                            mute=False)
            # ----Calculate meta_reward_final.----
            # TODO: Haven't found a better way to evaluate
            # ----option 1----
            #meta_reward = self.get_meta_reward(target_agent_performance)
            #for t in meta_transitions:
            #    t['reward'] = meta_reward
            #    replayBufferMeta.add(t)
            #logger.info(target_agent_performance)
            #logger.info(meta_reward)

            # ----option 2----
            reward_decay(meta_transitions, config.meta.gamma)
            for t in meta_transitions:
                replayBufferMeta.add(t)

            # ----Update controller using PPO.----
            for i in range(10):
                batch = replayBufferMeta.get_batch(replayBufferMeta.population)
                controller.update(batch, i)
            controller.sync_net()
            replayBufferMeta.clear()

            # ----Save contrller.----
            if ep_meta % config.meta.save_frequency == 0:
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

        # ----Old version, process a total episode at a time.----
        #env.reset_game_art()
        #env.set_goal_position()
        #env.set_init_position()
        #observation, reward, _ =\
        #    env.init_episode(config.emulator.display_flag)
        #state = preprocess(observation)

        ## ----Running one episode.----
        #total_reward = 0
        #for step in range(config.agent.total_steps - 1):
        #    action = agent.run_step(state, epsilon)
        #    observation, reward, alive = env.update(action)
        #    total_reward += reward
        #    next_state = preprocess(observation)
        #    transition = {'state': state,
        #                  'action': action,
        #                  'reward': reward,
        #                  'next_state': next_state}
        #    replayBuffer.add(transition)

        #    if replayBuffer.population > config.agent.batch_size:
        #        batch_size = config.agent.batch_size
        #    else:
        #        batch_size = replayBuffer.population
        #    batch = replayBuffer.get_batch(batch_size)
        #    agent.update(batch)

        #    state = next_state
        #    if not alive:
        #        break
        #if not mute:
        #    if alive:
        #        logger.info('failed')
        #    logger.info('total_reward this episode: {}'.format(total_reward))
        #    logger.info(step)
        # ----Old version, end.----

    def distill_agent_one_lesson(self, agent_t, agent_s, env, replayBuffer,
                                 epsilon, mute=False):
        # NOTE: We call M timesteps as an episode. Because the length of an
        # episode is unfixed so the training steps of each lesson is
        # different if we use a true episode as a lesson.

        config = self.config

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
            action = agent_t.run_step(state, epsilon)
            observation, reward, alive = env.update(action)
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
                if agent_s.update_steps % config.agent.synchronize_frequency == 0:
                    agent_s.sync_net()

            state = next_state
            if not alive:
                # ----One episode finished, start another.----
                env.reset_game_art()
                env.set_goal_position()
                env.set_init_position()
                obervation, _, _ = \
                    env.init_episode(config.emulator.display_flag)
                state = preprocess(observation)
        env.current_state = state

        # ----Old version, process a total episode at a time.----
        #env.reset_game_art()
        #env.set_goal_position()
        #env.set_init_position()
        #observation, reward, _ =\
        #    env.init_episode(config.emulator.display_flag)
        #state = preprocess(observation)

        #total_reward = 0
        ## ----Running one episode.----
        #for step in range(config.agent.total_steps):
        #    action = agent_t.run_step(state, epsilon)
        #    observation, reward, alive = env.update(action)
        #    total_reward += reward
        #    next_state = preprocess(observation)
        #    transition = {'state': state,
        #                  'action': action,
        #                  'reward': reward,
        #                  'next_state': next_state}
        #    replayBuffer.add(transition)

        #    if replayBuffer.population > config.agent.batch_size:
        #        batch_size = config.agent.batch_size
        #    else:
        #        batch_size = replayBuffer.population
        #    batch = replayBuffer.get_batch(batch_size)
        #    q_expert = agent_t.calc_q_value(batch['state'])
        #    batch['q_expert'] = q_expert
        #    agent_s.update_distill(batch)
        #    if agent_s.update_steps % config.agent.synchronize_frequency == 0:
        #        agent_s.sync_net()
        #    state = next_state
        #    if not alive:
        #        break
        #if not mute:
        #    if alive:
        #        logger.info('failed')
        #    logger.info('total_reward this episode: {}'.format(total_reward))
        # ----Old version, end.----

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
        return np.concatenate((pm.flatten() / 10, lp))

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
        #for i in range(len(self.agent_list)):
        #    reward += (new[i, i] - old[i, i])
        # TODO: only to check the meta model, remove later
        reward = new[2, 2] - old[2, 2]
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
        controller_ckpt = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/test/controller/'
        trainer.train_meta(save_model=False)
        #trainer.train_meta(save_model=False, load_model=controller_ckpt)
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


