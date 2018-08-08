from configparser import ConfigParser, ExtendedInterpolation
import json
import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
import utils
logger = utils.get_logger()

class Section(object):
    def __init__(self):
        pass

    def add(self, field, value):
        setattr(self, field, value)


class Parser(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), '{} not exists.'.format(config_path)
        self.config = ConfigParser(
            delimiters='=',
            interpolation=ExtendedInterpolation())
        self.config.read(config_path)
        self.data = self.read_data()
        self.task = self.read_task()
        self.meta = self.read_meta()

    def read_data(self):
        data = Section()
        data_dir = self.data_dir
        data.add('train_c_data_file', os.path.join(data_dir, self.config.get('data', 'train_c_data_file')))
        data.add('valid_c_data_file', os.path.join(data_dir, self.config.get('data', 'valid_c_data_file')))
        data.add('train_t_data_file', os.path.join(data_dir, self.config.get('data', 'train_t_data_file')))
        data.add('valid_t_data_file', os.path.join(data_dir, self.config.get('data', 'valid_t_data_file')))
        data.add('test_data_file', os.path.join(data_dir, self.config.get('data', 'test_data_file')))
        data.add('num_sample_valid_c', self.config.getint('data', 'num_sample_valid_c'))
        data.add('num_sample_train_c', self.config.getint('data', 'num_sample_train_c'))
        data.add('num_sample_valid_t', self.config.getint('data', 'num_sample_valid_t'))
        data.add('num_sample_train_t', self.config.getint('data', 'num_sample_train_t'))
        data.add('num_sample_test', self.config.getint('data', 'num_sample_test'))
        data.add('mean_noise', self.config.getfloat('data', 'mean_noise'))
        data.add('var_noise', self.config.getfloat('data', 'var_noise'))
        return data


    def read_task(self):
        task = Section()
        task.add('student_model_name', self.config.get('task', 'student_model_name'))
        task.add('batch_size', self.config.getint('task', 'batch_size_task'))
        task.add('dim_input', self.config.getint('task', 'dim_input_task'))
        task.add('dim_hidden', self.config.getint('task', 'dim_hidden_task'))
        task.add('dim_output', self.config.getint('task', 'dim_output_task'))
        task.add('lr', self.config.getfloat('task', 'lr_task'))
        task.add('valid_frequency', self.config.getint('task', 'valid_frequency_task'))
        task.add('max_endurance', self.config.getint('task', 'max_endurance_task'))
        task.add('max_training_step', self.config.getint('task', 'max_training_step_task'))
        task.add('lambda1', self.config.getfloat('task', 'lambda1_task'))
        task.add('lambda2', self.config.getfloat('task', 'lambda2_task'))
        return task


    def read_meta(self):
        meta = Section()
        meta.add('history_len', self.config.getint('meta', 'history_len_meta'))
        meta.add('total_episodes', self.config.getint('meta', 'total_episodes_meta'))
        meta.add('buffer_size', self.config.getint('meta', 'buffer_size_meta'))
        meta.add('ema_decay_state', self.config.getfloat('meta', 'ema_decay_state'))
        meta.add('dim_a', self.config.getint('meta', 'dim_a_meta'))
        meta.add('dim_h', self.config.getint('meta', 'dim_h_meta'))
        meta.add('dim_s', self.config.getint('meta', 'dim_s_meta'))
        meta.add('cliprange', self.config.getfloat('meta', 'cliprange_meta'))
        meta.add('controller', self.config.get('meta', 'controller'))
        meta.add('batch_size', self.config.getint('meta', 'batch_size_meta'))
        meta.add('lr', self.config.getfloat('meta', 'lr_meta'))
        meta.add('logits_scale', self.config.getfloat('meta', 'logits_scale_meta'))
        meta.add('save_frequency', self.config.getint('meta', 'save_frequency_meta'))
        meta.add('reward_c', self.config.get('meta', 'reward_c'))
        meta.add('gamma', self.config.getfloat('meta', 'gamma_meta'))
        meta.add('n_parallel_actor', self.config.getint('meta', 'n_parallel_actor'))
        meta.add('entropy_bonus_beta', self.config.getfloat('meta', 'entropy_bonus_beta_meta'))
        meta.add('one_step_td', self.config.getboolean('meta', 'one_step_td'))
        meta.add('warmup_steps', self.config.getint('meta', 'warmup_steps_meta'))
        meta.add('reward_max_value', self.config.getfloat('meta', 'reward_max_value'))
        meta.add('reward_baseline_decay', self.config.getfloat('meta', 'reward_baseline_decay'))
        meta.add('max_endurance_meta', self.config.getint('meta', 'max_endurance_meta'))

        return meta

    def print_config(self):
        for key_sec, sec in self.config.items():
            logger.info('[{}]'.format(key_sec))
            for key, value in sec.items():
                logger.info('{}:: {}'.format(key, value))

    #######
    @property
    def exp_dir(self):
        return os.path.expanduser(self.config.get('env', 'exp_dir'))

    @property
    def model_dir(self):
        if socket.gethostname() == 'Luna-Desktop':
            return os.path.expanduser(self.config.get('env', 'model_dir2'))
        else:
            return os.path.expanduser(self.config.get('env', 'model_dir1'))

    @property
    def data_dir(self):
        return os.path.expanduser(self.config.get('env', 'data_dir'))

    @property
    def save_images_dir(self):
        return os.path.expanduser(self.config.get('env', 'save_images_dir'))


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'config/gridworld.cfg')
    config = Parser(config_path)
    print(config.agent.dim_a)
