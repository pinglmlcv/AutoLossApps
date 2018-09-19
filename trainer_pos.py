'''
The module for training part of speech tagging task.
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

from models.s2s import rnn_attention
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
        self.model = rnn_attention.RNNAttention(config,
                                                self.sess,
                                                mode,
                                                exp_name=exp_name,
                                                logger=logger)
        self.train_datasets = Dataset()
        self.valid_datasets = Dataset()
        self.test_datasets = Dataset()
        self.train_datasets.load_json(config.train_data_file)
        self.valid_datasets.load_json(config.valid_data_file)
        self.test_datasets.load_json(config.test_data_file)

    def train(self, load_model=None, save_model=False):
        config = self.config
        batch_size = config.batch_size
        start_time = time.time()
        model = self.model
        model.init_parameters()
        if load_model:
            model.load_model()
        loss = 0.0
        acc = 0.0

        # Training loop
        logger.info('Start training...')
        for step in range(config.max_training_steps):
            samples = self.train_datasets.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)

            step_loss, step_acc, summary = model.train(inputs, inputs_len,
                                                       targets, targets_len,
                                                       return_acc=True)
            loss += float(step_loss) / config.display_frequency
            acc += float(step_acc) / config.display_frequency
            real_step = model.global_step.eval()
            if real_step % config.display_frequency == 0:
                time_elapsed = time.time() - start_time
                step_time = time_elapsed / config.display_frequency
                logger.info('Step: {}, Loss: {}, Acc: {}'.\
                            format(real_step, loss, acc))
                loss = 0
                acc = 0
                start_time = time.time()

            if real_step % config.valid_frequency == 0:
                logger.info('Validation step:')
                valid_loss, valid_acc = self.valid()
                logger.info('Valid loss: {}, Valid acc: {}'.\
                            format(valid_loss, valid_acc))

            if real_step % config.save_frequency == 0:
                logger.info('Saving the model...')
                model.save_model(real_step)

    def valid(self):
        config = self.config
        batch_size = config.batch_size
        valid_loss = 0.0
        valid_acc = 0.0
        valid_count = 0
        dataset = self.valid_datasets
        dataset.reset(shuffle=True)
        while dataset.epochs_completed < 1:
            samples = dataset.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len, targets, targets_len = prepare_train_batch(
                inputs, targets, config.max_seq_length)
            batch_loss, batch_acc, summary = self.model.eval(inputs, inputs_len,
                                                             targets, targets_len,
                                                             return_acc=True)
            valid_loss += batch_loss * batch_size
            valid_acc += batch_acc * batch_size
            valid_count += batch_size

        valid_loss = valid_loss / valid_count
        valid_acc = valid_acc / valid_count
        return valid_loss, valid_acc

    def inference(self, load_model):
        config = self.config
        batch_size = config.batch_size
        model = self.model
        model.load_model(load_model)
        # Inference
        logger.info('Start inferencing...')
        dataset = self.valid_datasets
        dataset.reset()
        correct_prediction = 0
        total_tokens = 0
        while dataset.epochs_completed < 1:
            samples = dataset.next_batch(batch_size)
            inputs = samples['input']
            targets = samples['target']
            inputs, inputs_len = prepare_batch(inputs)

            # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
            # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
            predicts = model.inference(encoder_inputs=inputs,
                                       encoder_inputs_length=inputs_len)
            predicts = data_utils.unpaddle(predicts[:,:,0])
            acc, count = metrics.accuracy_token(targets, predicts)
            correct_prediction += acc * count
            total_tokens += count
        accuracy = correct_prediction / total_tokens
        logger.info('Accuracy on testset: {}'.format(accuracy))



if __name__ == '__main__':
    argv = sys.argv
    # ----Parsing config file.----
    logger.info('Machine: {}'.format(socket.gethostname()))
    cfg_dir = os.path.join(root_path, 'config/posRNNAttention/')
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
        load_model = os.path.join(config.model_dir, argv[1])
        trainer.inference(load_model)
    elif argv[2] == 'baseline_multi':
        # ----Baseline----
        logger.info('BASELINE')
    elif argv[2] == 'generate':
        # ----Generating----
        logger.info('GENERATE')


