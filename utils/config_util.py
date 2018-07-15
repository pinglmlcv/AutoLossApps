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
        self.agent = self.read_agent()
        self.meta = self.read_meta()
        self.emulator = self.read_emulator()


    def read_agent(self):
        agent = Section()
        agent.add('dim_a', self.config.getint('agent', 'dim_a_agent'))
        agent.add('dim_s_h', self.config.getint('agent', 'dim_s_h_agent'))
        agent.add('dim_s_w', self.config.getint('agent', 'dim_s_w_agent'))
        agent.add('dim_s_c', self.config.getint('agent', 'dim_s_c_agent'))
        agent.add('distill_mode', self.config.get('agent', 'distill_mode'))
        agent.add('kl_temp', self.config.getfloat('agent', 'kl_temp'))
        agent.add('c_kl', self.config.getfloat('agent', 'c_kl_agent'))
        agent.add('c_ent', self.config.getfloat('agent', 'c_ent_agent'))
        agent.add('c_distill', self.config.getfloat('agent', 'c_distill_agent'))
        agent.add('lr', self.config.getfloat('agent', 'lr_agent'))
        agent.add('total_episodes', self.config.getint('agent', 'total_episodes_agent'))
        agent.add('total_episodes_test', self.config.getint('agent', 'total_episodes_test_agent'))
        agent.add('total_steps', self.config.getint('agent', 'total_steps_agent'))
        agent.add('gamma', self.config.getfloat('agent', 'gamma_agent'))
        agent.add('epsilon_start', self.config.getfloat('agent', 'epsilon_start'))
        agent.add('epsilon_end', self.config.getfloat('agent', 'epsilon_end'))
        agent.add('epsilon_decay_steps', self.config.getint('agent', 'epsilon_decay_steps'))
        agent.add('buffer_size', self.config.getint('agent', 'buffer_size_agent'))
        agent.add('batch_size', self.config.getint('agent', 'batch_size_agent'))
        agent.add('synchronize_frequency', self.config.getint('agent', 'synchronize_frequency_agent'))
        agent.add('valid_frequency', self.config.getint('agent', 'valid_frequency_agent'))
        agent.add('save_frequency', self.config.getint('agent', 'save_frequency_agent'))
        agent.add('mute', self.config.getboolean('agent', 'mute_agent'))
        agent.add('lesson_length', self.config.getint('agent', 'lesson_length'))
        agent.add('alpha_reg', self.config.getfloat('agent', 'alpha_reg_agent'))
        agent.add('beta_reg', self.config.getfloat('agent', 'beta_reg_agent'))

        return agent

    def read_meta(self):
        meta = Section()
        meta.add('total_episodes', self.config.getint('meta', 'total_episodes_meta'))
        meta.add('buffer_size', self.config.getint('meta', 'buffer_size_meta'))
        meta.add('ema_decay_state', self.config.getfloat('meta', 'ema_decay_state'))
        meta.add('ema_decay_auc_baseline', self.config.getfloat('meta', 'ema_decay_auc_baseline'))
        meta.add('dim_a', self.config.getint('meta', 'dim_a_meta'))
        meta.add('dim_h', self.config.getint('meta', 'dim_h_meta'))
        meta.add('dim_s', self.config.getint('meta', 'dim_s_meta'))
        meta.add('cliprange', self.config.getfloat('meta', 'cliprange_meta'))
        meta.add('controller', self.config.get('meta', 'controller'))
        meta.add('batch_size', self.config.getint('meta', 'batch_size_meta'))
        meta.add('lr', self.config.getfloat('meta', 'lr_meta'))
        meta.add('logits_scale', self.config.getfloat('meta', 'logits_scale_meta'))
        meta.add('save_frequency', self.config.getint('meta', 'save_frequency_meta'))
        meta.add('reward_strategy', self.config.get('meta', 'reward_strategy_meta'))
        meta.add('gamma', self.config.getfloat('meta', 'gamma_meta'))
        meta.add('n_parallel_actor', self.config.getint('meta', 'n_parallel_actor'))
        meta.add('entropy_bonus_beta', self.config.getfloat('meta', 'entropy_bonus_beta_meta'))
        meta.add('one_step_td', self.config.getboolean('meta', 'one_step_td'))
        meta.add('warmup_steps', self.config.getint('meta', 'warmup_steps_meta'))

        return meta

    def read_emulator(self):
        emlt = Section()
        emlt.add('display_flag', self.config.getboolean('emulator', 'display_flag'))

        return emlt


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
    def save_images_dir(self):
        return os.path.expanduser(self.config.get('env', 'save_images_dir'))

    @property
    def train_data_file(self):
        train_data_file = self.config.get('data', 'train_data_file')
        return os.path.join(self.data_dir, train_data_file)

    @property
    def valid_data_file(self):
        valid_data_file = self.config.get('data', 'valid_data_file')
        return os.path.join(self.data_dir, valid_data_file)

    @property
    def train_stud_data_file(self):
        train_stud_data_file = self.config.get('data', 'train_stud_data_file')
        return os.path.join(self.data_dir, train_stud_data_file)

    @property
    def test_data_file(self):
        test_data_file = self.config.get('data', 'test_data_file')
        return os.path.join(self.data_dir, test_data_file)


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'config/gridworld.cfg')
    config = Parser(config_path)
    print(config.agent.dim_a)
