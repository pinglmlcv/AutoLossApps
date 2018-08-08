""" Generate toy dataset for linear regression with L1, L2 regularization """

# __author__ == 'Haowen Xu'
# __data__ == '04_07_2018'

import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
import numpy as np
import json
import utils
from utils import config_reg_util

config_path = os.path.join(root_path, 'config/regression.cfg')
config = config_reg_util.Parser(config_path)
dim = config.task.dim_input
mean_noise = config.data.mean_noise
var_noise = config.data.var_noise
np.random.seed(1)


def linear_func(x, w):
    return np.dot(x, w)

def load():
    with open(data_train, 'rb') as f:
        data = np.load(f)
        print(data)

def make_data_file(w, num, data_file):
    with open(data_file, 'wb') as f:
        data = []
        for i in range(num):
            x = 10 * (np.random.rand(dim) - 0.5)
            y = linear_func(x, w) + np.random.normal(loc=mean_noise,
                                                     scale=var_noise)
            data.append({'x': x, 'y': y})
        np.save(f, data)
        f.close()

def main():
    # generator training dataset
    w = np.random.rand(dim) - 0.5
    #w = np.array([0.1, 0.5, 0.5, -0.1, 0.2, -0.2, 0.3, 0.1])
    print('weight: ', w)

    conf = config.data
    make_data_file(w, conf.num_sample_train_c, conf.train_c_data_file)
    make_data_file(w, conf.num_sample_valid_c, conf.valid_c_data_file)
    make_data_file(w, conf.num_sample_train_t, conf.train_t_data_file)
    make_data_file(w, conf.num_sample_valid_t, conf.valid_t_data_file)
    make_data_file(w, conf.num_sample_test, conf.test_data_file)


if __name__ == '__main__':
    main()
