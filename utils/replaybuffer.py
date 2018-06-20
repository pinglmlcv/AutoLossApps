# __Author__ == 'Haowen Xu"
# __Date__ == "06-15-2018"

import numpy as np
import random
import scipy
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.population = 0
        self.transition_buffer = deque(maxlen=buffer_size)

    def add(self, transition):
        self.transition_buffer.append(transition)
        if self.population < self.buffer_size:
            self.population += 1

    def clear(self):
        self.population = 0
        self.transition_buffer.clear()

    def get_batch(self, batch_size):
        if self.population < batch_size:
            raise Exception('buffer has less data point than'
                            'batchsize {}'.format(batch_size))
        batch = random.sample(self.transition_buffer, batch_size)
        state = []
        reward = []
        action = []
        next_state = []
        for t in batch:
            state.append(t['state'])
            reward.append(t['reward'])
            action.append(t['action'])
            next_state.append(t['next_state'])
        batch = {'state': state,
                 'reward': reward,
                 'action': action,
                 'next_state': next_state}
        return batch

