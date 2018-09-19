import tensorflow as tf
import numpy as np

from models.basic_model import Basic_model
import utils
logger = utils.get_logger()

class FixedController(Basic_model):
    def __init__(self, config, sess, exp_name='fixed_controller'):
        super(FixedController, self).__init__(config, sess, exp_name)
        self.init = tf.constant([1])

    def run_step(self, state, ep, epsilon=0):
        #return 2
        p = np.array([102736, 23999, 40175])
        p = p / np.sum(p)
        action = np.random.choice(len(p), p=p)
        return action
