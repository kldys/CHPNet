
import scipy.io as io
import random


class DefaultConfig(object):
    scaling_factor = 2
    LR_size = 32
    HR_size = LR_size*scaling_factor

    data_root = 'data_root'
    label_root = 'label_root'

    num_data = 4000

    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 0  # how many workers for loading data

    max_epoch = 30
    lr = 0.0005  # initial learning rate
    lr_decay = 0.5

    load_model_path = 'model_path'
    save_model_path = 'model_path'
    teacher_model_path='teacher_model_path'


opt = DefaultConfig()

























