import os, sys
import socket
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, root_path)
from utils.config_util import Section

class Config():
    def __init__(self):
        self.hostname = socket.gethostname()
        # Tasks definition
        self.task_names = ['mt', 'ner', 'pos'] # The first task is considered as the primary task
        self.task_name_id_dict = {'mt': 0, 'ner': 1, 'pos': 2}

        # Environment & Path
        self.train_data_files = [os.path.join(root_path, 'Data/nlp/{}/{}_train.json'.format(task, task))
                                 for task in self.task_names]
        self.valid_data_files = [os.path.join(root_path, 'Data/nlp/{}/{}_valid.json'.format(task, task))
                                 for task in self.task_names]
        self.test_data_files = [os.path.join(root_path, 'Data/nlp/{}/{}_test.json'.format(task, task))
                                for task in self.task_names]
        if self.hostname == 'jungpu4':
            self.model_dir = '/home/haowen/saved/AutoLossApps/saved_models'
        elif self.hostname == 'Luna-Desktop':
            self.model_dir = '/media/haowen/AutoLossApps/saved_models'


        # Data
        self.max_seq_length = 60
        self.num_encoder_symbols = 10000
        self.num_decoder_symbols = [10000, 29, 61]

        # Model Architecture
        self.cell_type = 'LSTM'
        self.attention_type = 'luong'
        self.attn_input_feeding = False
        self.use_dropout = True
        self.use_residual = True

        self.embedding_size = 128

        self.encoder_hidden_units = 256
        self.encoder_depth = 2

        self.attn_hidden_units = 256
        self.decoder_hidden_units = 256
        self.decoder_depth = 2

        # Training
        self.max_training_steps = 100000
        self.batch_size = 128
        self.optimizer = 'adam'
        self.learning_rate = 0.0002
        self.max_gradient_norm = 1.0
        self.keep_prob = 0.7

        self.display_frequency = 100
        self.valid_frequency = 500
        self.save_frequency = 500

        # Decoding
        self.max_decode_step = 70
        self.beam_width = 5

        # Controller
        self.controller_type = 'Fixed'
        self.buffer_size = 20000

        # Task
        self.history_len_task = 10

    def print_config(self, logger):
        for key, value in vars(self).items():
            logger.info('{}:: {}'.format(key, value))

def get_cfg():
    return Config()

if __name__ == '__main__':
    print(root_path)
