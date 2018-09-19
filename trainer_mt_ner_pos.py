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

from models.s2s import rnn_attention_multi_heads
from models import controllers
from dataio.dataset import Dataset
import utils
from utils import config_util
from utils import data_utils
from utils.data_utils import prepare_batch
from utils.data_utils import prepare_train_batch
from utils import metrics

root_path = os.path.dirname(os.path.realpath(__file__))
logger = utils.get_logger()

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
        self.controller = controllers.FixedController(config, self.sess)

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
        model.init_parameters()
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


