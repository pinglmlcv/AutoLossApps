"""
Basic model object
New features:
    Using tf.data.dataset as input pipline.
    Hyperpapameters are defined directly in config object, remove cfg module.
__Author__ == 'Haowen Xu'
__Data__ == '09-11-2018'
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import time
import os

from .. import utils

logger = utils.get_logger()

class Basic_model():
    def __init__(self, config, sess, exp_name='new_exp'):
        self.config = config
        self.exp_name = exp_name
        self.sess = sess
        self.checkpoint_dir = os.path.join(self.config.model_dir, exp_name)

    def reset(self):
        raise NotImplementedError

    def _build_placeholder(self):
        raise NotImplementedError

    def _build_graph(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def load_model(self, checkpoint_dir=None, ckpt_num=None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if not ckpt_num:
            if ckpt and ckpt.model_checkpoint_path:
                model_checkpoint_path = ckpt.model_checkpoint_path
            else:
                raise Exception('Invalid checkpoint_dir: ' + checkpoint_dir)
        else:
            model_checkpoint_path = None
            for cp in ckpt.all_model_checkpoint_paths:
                if cp.split('-')[-1] == str(ckpt_num):
                    model_checkpoint_path = cp
            if not model_checkpoint_path:
                raise Exception('Ckpt num {} not found!'.format(ckpt_num))
        logger.info('Loading model from: ' + model_checkpoint_path)
        self.saver.restore(self.sess, model_checkpoint_path)

    def save_model(self, step):
        task_name = self.exp_name
        model_dir = self.config.model_dir
        task_dir = os.path.join(model_dir, task_name)
        self.task_dir = task_dir
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)
        save_path = os.path.join(task_dir, 'model')
        logger.info('Save model at {}'.format(save_path))
        self.saver.save(self.sess, save_path, global_step=step)

    def print_weights(self, tvars=None):
        if not tvars:
            for tvar in self.tvars:
                logger.info('===begin===')
                logger.info(tvar)
                logger.info(self.sess.run(tvar))
                logger.info('===end===')
        else:
            for tvar in tvars:
                logger.info('===begin===')
                logger.info(tvar)
                logger.info(self.sess.run(tvar))
                logger.info('===end===')

    def get_grads_magnitude(self, grads):
        v = []
        for grad in grads:
            v.append(np.reshape(grad, [-1]))
        v = np.concatenate(v)
        return np.linalg.norm(v) / np.sqrt(v.shape[0])

    def initialize_weights(self):
        self.sess.run(self.init)

