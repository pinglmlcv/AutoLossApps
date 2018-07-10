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
import gym
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

    # TODO Set arbitory supervision to see if the critic network can predict
    #for i in range(len(transitions)):
    #    s = transitions[i]['state'][:, :, 1]
    #    p = np.where(s == np.max(s))
    #    value = p[0] + p[1]
    #    transitions[i]['target_value'] = value[0]

    #for i in range(len(transitions)):
    #    print(transitions[i]['target_value'])


class MlpPPO(tdppo_controller.BasePPO):
    def __init__(self, config, sess, exp_name='MlpPPO'):
        super(MlpPPO, self).__init__(config, sess, exp_name)
        with tf.variable_scope(exp_name):
            self._build_placeholder()
            self._build_graph()

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
                trainable=trainable,
                scope='fc1')

            value = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                trainable=trainable,
                scope='fc2')

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '{}/{}'.format(self.exp_name, scope))
            return value, param

    def run_step(self, states, epsilon=0):
        if randomo.random() < epsilon:
            return random.randint(0, self.config.agent.dim_a - 1), 0
        else:
            pi = self.sess.run(self.pi, {self.state: state})[0]
            action = np.random.choice(dim_a, 1, p=p1)[0]
            return action, pi


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
        elif config.meta.controller == 'CNNPPO':
            self.controller = tdppo_controller.CNNPPO(config, self.sess,
                exp_name=exp_name+'/controller')
        else:
            raise Exception('Invalid controller name')

        #self.env = two_rooms.Env2Rooms(config, default_goal=(3, 3),
        #                               default_init=(7, 5))
        #self.env = two_rooms.Env2Rooms(config, default_goal=(3, 3))
        self.env = gym.make('Acrobot-v1')
        self.config.agent.dim_a = self.env.action_space.n
        self.config.agent.dim_s = self.env.observation_space.shape[0]

    def train(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        self.no_reward_since = 0
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBuffer = replaybuffer.ReplayBuffer(config.agent.buffer_size,
                                                 keys=keys)

        epsilons = np.linspace(config.agent.epsilon_start,
                               config.agent.epsilon_end,
                               config.agent.epsilon_decay_steps)

        # ----Initialize performance matrix.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)

        for ep in range(config.agent.total_episodes):
            logger.info('######## episodes {} #########'.format(ep))
            start_time = time.time()
            epsilon = epsilons[min(ep, config.agent.epsilon_decay_steps-1)]
            self.train_agent_one_lesson_gym(controller,
                                            self.env,
                                            replayBuffer,
                                            epsilon,
                                            mute=config.agent.mute)

            if ep % 5 == 0:
                logger.info('--test ep{}--'.format(ep))
                self.test_agent(self.controller,
                                self.env,
                                num_episodes=100,
                                mute=False)
                logger.info('----')

            if ep % 5 == 0:
                controller.save_model(ep)

        # ----End of a meta episode.----
        logger.info('running time: {}'.format(time.time() - start_time))
        logger.info('#################')

    def train_agent_one_lesson(self, agent, env, replayBuffer, epsilon,
                               mute=False):
        transitions = []
        config = self.config
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBuffer_critic = replaybuffer.ReplayBuffer(config.agent.buffer_size,
                                                        keys=keys)

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
            action, pi = agent.run_step(state, epsilon)
            value = agent.get_value([state])
            #logger.info(pi)
            #logger.info(value)
            observation, reward, alive = env.update(action, display=str(value))
            next_state = preprocess(observation)
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'target_value': reward,
                          'next_state': next_state}
            transitions.append(transition)
            state = next_state
            if not alive:
                # ----One episode finished, start another.----
                env.reset_game_art()
                env.set_goal_position()
                env.set_init_position()
                observation, _, _ = \
                    env.init_episode(config.emulator.display_flag)
                state = preprocess(observation)

                reward_decay(transitions, config.agent.gamma)
                for transition in transitions:
                    replayBuffer_critic.add(transition)
                    replayBuffer.add(transition)
                transitions = []

        for i in range(10):
            # update actor
            batch_size = min(replayBuffer.population, config.agent.batch_size)
            batch = replayBuffer.get_batch(batch_size)
            next_value = agent.get_value(batch['next_state'])
            batch['next_value'] = next_value[:, 0]
            agent.update_actor(batch)

            # update critic
            batch_size = min(replayBuffer_critic.population,
                             config.agent.batch_size)
            batch = replayBuffer_critic.get_batch(batch_size)
            agent.update_critic(batch)

        agent.sync_net()
        if not config.meta.one_step_td:
            replayBuffer.clear()

        env.current_state = state

    def train_agent_one_lesson_gym(self, agent, env, replayBuffer, epsilon,
                               mute=False):
        transitions = []
        config = self.config
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBuffer_critic = replaybuffer.ReplayBuffer(config.agent.buffer_size,
                                                        keys=keys)

        # ----Lesson version.----
        state = env.reset()
        total_reward = 0

        for step in range(config.agent.lesson_length):
            env.render()
            action, _ = agent.run_step(state[np.newaxis,:], epsilon)
            next_state, reward, done, _ = env.update(action)

            total_rewards += reward
            reward = 5.0 if done else -0.1
            transition = {'state': state,
                          'action': action,
                          'reward': reward,
                          'target_value': reward,
                          'next_state': next_state}

            transitions.append(transition)
            state = next_state
            if done:
                reward_decay(transitions, config.agent.gamma)
                for transition in transitions:
                    replayBuffer.add(transition)
                    replayBuffer_critic.add(transition)
                break
        if total_reward <= -500:
            self.no_reward_since += 1
            if self.no_reward_since >= 5:
                print('Resetting model... start anew!')
                agent.initialize_weights()
                self.no_reward_since = 0
                return 0
        else:
            no_reward_since = 0

        for i in range(10):
            # update actor
            batch_size = min(replayBuffer.population, config.agent.batch_size)
            batch = replayBuffer.get_batch(batch_size)
            next_value = agent.get_value(batch['next_state'])
            batch['next_value'] = next_value[:, 0]
            agent.update_actor(batch)

            # update critic
            batch_size = min(replayBuffer_critic.population,
                             config.agent.batch_size)
            batch = replayBuffer_critic.get_batch(batch_size)
            agent.update_critic(batch)

        agent.sync_net()
        if not config.meta.one_step_td:
            replayBuffer.clear()

        print(total_rewards)


    def test(self, load_model, ckpt_num=None):
        config = self.config
        agent = self.controller
        env = self.env

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
                action, _ = agent.run_step(state, epsilon)
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
        load_model = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/'\
            'tdppo2/controller'
        #trainer.train(save_model=True, load_model=load_model)
        trainer.train(save_model=True)
    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        agent_dir = '/datasets/BigLearning/haowen/AutoLossApps/saved_models/'\
            'tdppo/controller'.format(argv[1])
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


