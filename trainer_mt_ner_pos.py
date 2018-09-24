'''
The module for mulit-task training of machine translation, named entity
recognition and part of speech.
# __Author__ == 'Haowen Xu'
# __Date__ == '09-12-2018'
'''

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

from models.s2s import rnn_attention_multi_heads
from models import controllers
from dataio.dataset import Dataset
import utils
from utils import config_util
from utils import data_utils
from utils.data_utils import prepare_batch
from utils.data_utils import prepare_train_batch
from utils import metrics
from utils import replaybuffer

root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

class MyMlpPPO(controllers.MlpPPO):
    def build_actor_net(self, scope, trainable):
        pass


class Trainer():
    """ A class to wrap training code. """
    def __init__(self, config, mode='TRAIN', exp_name=None):
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
        self.model = rnn_attention_multi_heads.RNNAttentionMultiHeads(
            config,
            self.sess,
            mode,
            exp_name=exp_name,
            logger=logger
        )
        if config.controller_type == 'Fixed':
            self.controller = controllers.FixedController(config, self.sess)
        elif config.controller_type == 'MlpPPO':
            self.controller = MyMlpPPO(config, self.sess)

        self.train_datasets = []
        self.valid_datasets = []
        self.test_datasets = []
        for task_num, task in enumerate(config.task_names):
            train_datasets = Dataset()
            valid_datasets = Dataset()
            test_datasets = Dataset()
            train_datasets.load_json(config.train_data_files[task_num])
            valid_datasets.load_json(config.valid_data_files[task_num])
            test_datasets.load_json(config.test_data_files[task_num])
            self.train_datasets.append(train_datasets)
            self.valid_datasets.append(valid_datasets)
            self.test_datasets.append(test_datasets)

    def train(self, load_model=None, save_model=False):
        config = self.config
        batch_size = config.batch_size
        start_time = time.time()
        model = self.model
        controller = self.controller
        model.init_weights()
        if load_model:
            model.load_model()
        loss = 0.0
        acc = 0.0
        main_task_count = 0

        # Training loop
        logger.info('Start training...')
        for step in range(config.max_training_steps):
            task_to_train = controller.run_step(0, 0)
            samples = self.train_datasets[task_to_train].next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)

            step_loss, step_acc = model.train(task_to_train,
                                                       inputs, inputs_len,
                                                       targets, targets_len,
                                                       return_acc=True)
            if task_to_train == 0:
                loss += float(step_loss)
                acc += float(step_acc)
                main_task_count += 1
            real_step = model.global_step.eval()
            if real_step % config.display_frequency == 0:
                time_elapsed = time.time() - start_time
                step_time = time_elapsed / config.display_frequency
                loss = loss / main_task_count
                acc = acc / main_task_count
                logger.info('Step: {}, Loss: {}, Acc: {}'.\
                            format(real_step, loss, acc))
                loss = 0.0
                acc = 0.0
                main_task_count = 0
                start_time = time.time()

            if real_step % config.valid_frequency == 0:
                logger.info('Validation step:')
                valid_loss, valid_acc = self.valid(0)
                logger.info('Valid loss: {}, Valid acc: {}'.\
                            format(valid_loss, valid_acc))

            if real_step % config.save_frequency == 0:
                logger.info('Saving the model...')
                model.save_model(real_step)

    def train_meta(self, load_model=None, save_model=False):
        config = self.config
        controller = self.controller
        model = self.model
        keys = ['state', 'next_state', 'reward', 'action', 'target_value']
        replayBufferMeta = replaybuffer.ReplayBuffer(
            config.buffer_size, keys=keys)
        meta_training_history = deque(maxlen=10)
        epsilons = np.linspace(config.epsilon_start_meta,
                               config.epsilon_end_meta,
                               config.epsilon_decay_steps_meta)

        # ----Initialize controller.----
        controller.initialize_weights()
        if load_model:
            controller.load_model(load_model)

        # ----Start meta loop.----
        for ep_meta in range(config.max_episodes_meta):
            logger.info('######## meta_episodes {} #########'.format(ep_meta))
            start_time = time.time()
            # ----Initialize task model.----
            model.initialize_weights()
            model.reset()
            history_train_loss = []
            history_train_acc = []
            history_valid_loss = []
            history_valid_acc = []
            history_len_task = config.history_len_task
            for i in range(len(config.task_names)):
                history_train_loss.append(deque(maxlen=history_len_task))
                history_train_acc.append(deque(maxlen=history_len_task))
                history_valid_loss.append(deque(maxlen=history_len_task))
                history_valid_acc.append(deque(maxlen=history_len_task))

            meta_state = self.get_meta_state(history_train_loss,
                                             history_train_acc,
                                             history_valid_loss,
                                             history_valid_acc,
                                             )
            main_task_count = 0
            for ep in range(config.max_training_steps):
                epsilon = epsilons[min(ep, config.meta_epsilon_decay_steps)]
                meta_action = controller.run_step(meta_state, ep, epsilon)
                task_to_train = meta_action
                logger.info('task_to_train: {}'.format(task_to_train))
                samples = self.train_datasets[task_to_train].next_batch(batch_size)
                inputs = samples['input']
                targets = samples['target']
                inputs, inputs_len, targets, targets_len = prepare_train_batch(
                    inputs, targets, config.max_seq_length)

                step_loss, step_acc = model.train(task_to_train,
                                                  inputs, inputs_len,
                                                  targets, targets_len,
                                                  return_acc=True)
                history_train_loss[task_to_train].append(step_loss)
                history_train_acc[task_to_train].append(step_acc)
                if step % config.valid_frequency == 0:
                    for i in range(len(config.task_names)):
                        valid_loss, valid_acc = self.valid(i)
                        history_valid_loss[i].append(valid_loss)
                        history_valid_acc[i].append(valid_acc)

                meta_reward = self.get_meta_reward_real_time(history_train_acc[0])
                meta_state_new = self.get_meta_state(history_train_loss,
                                                     history_train_acc,
                                                     history_valid_loss,
                                                     history_valid_acc
                                                     )
                transition = {'state': meta_state,
                              'action': meta_action,
                              'reward': meta_reward,
                              'next_state': meta_state_new,
                              'target_value': None}
                meta_transitions.append(transition)
                meta_state = meta_state_new

                if self.check_terminate():
                    break

                pass





    def valid(self, task_to_eval):
        config = self.config
        batch_size = config.batch_size
        valid_loss = 0.0
        valid_acc = 0.0
        valid_count = 0
        dataset = self.valid_datasets[task_to_eval]
        dataset.reset(shuffle=True)
        while dataset.epochs_completed < 1:
            samples = dataset.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)
            batch_loss, batch_acc = self.model.eval(task_to_eval,
                                                             inputs, inputs_len,
                                                             targets, targets_len,
                                                             return_acc=True)
            valid_loss += batch_loss * batch_size
            valid_acc += batch_acc * batch_size
            valid_count += batch_size

        valid_loss = valid_loss / valid_count
        valid_acc = valid_acc / valid_count
        return valid_loss, valid_acc


if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info('Machine: {}'.format(socket.gethostname()))
    cfg_dir = os.path.join(root_path, 'config/mtNerPosRNNAttention/')
    config = config_util.load_config(cfg_dir)
    config.print_config(logger)

    # ----Instantiate a trainer object.----

    if argv[2] == 'train':
        # ----Training----
        logger.info('TRAIN')
        trainer = Trainer(config, exp_name=argv[1])
        trainer.train()

    elif argv[2] == 'test':
        ## ----Testing----
        logger.info('TEST')
        trainer = Trainer(config, mode='DECODE', exp_name=argv[1])
        # TODO
    elif argv[2] == 'baseline_multi':
        # ----Baseline----
        logger.info('BASELINE')
    elif argv[2] == 'generate':
        # ----Generating----
        logger.info('GENERATE')
