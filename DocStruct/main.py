import argparse
import json
import logging
import random
import shutil
import warnings
from datetime import datetime

import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

import models.evaluation as eval
import models.loss as loss
from models.baseline import Baseline
from models.layoutlm import LayoutLM
from models.model import JointModel
from train import train
from utils.dataset import load_data
from utils.linklink_utils import *

warnings.filterwarnings('ignore')


def set_random_seed(seed_value=123456):
    """ set random seed for recurrence
    include torch torch.cuda numpy random """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def set_logger(mode):
    """ setting console log and file log (only train mode) of
    root logger with info level, copying config file to same
    directory for recurrence, directory path: log/m-d-h-M-feature
    (remove directory if it exists), then use logging.debug /
    logging.info / ... in all python package
    Args:
        mode: run mode (train or test)
    Return:
        logger_path: log directory (for saving model and other data)
    """
    log_dir = config.run.log_dir

    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s] [Line %(lineno)d] %(message)s'
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    log = logging.getLogger()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    log.addHandler(stream_handler)

    time = datetime.now().strftime("%b%d-%H%M%S")
    data_tag = config.dataset.tag
    sp_tag = '' if config.run.tag is None else f'-{config.run.tag}'
    feature_tag = ''.join([f[0] for f in config.model.feature])
    tag = f'{data_tag}-{config.model.name}-{feature_tag}-{config.model.fusion}'
    logger_path = '{}-{}'.format(tag, time) + sp_tag

    if rank == 0:
        log.setLevel(logging.INFO)
        if mode == 'train':
            os.makedirs(os.path.join(log_dir, logger_path))
            shutil.copy(args.config, os.path.join(log_dir, logger_path, 'config.yaml'))
        elif mode == 'test':
            if not os.path.exists('log/tmp/'):
                os.makedirs('log/tmp/')
    else:
        log.setLevel(logging.WARNING)

    link.barrier()

    if mode == 'train':
        if rank == 0:
            file_path = os.path.join(log_dir, logger_path, 'info.log')
            file_handler = logging.FileHandler(file_path, mode='w')
            file_handler.setFormatter(log_format)
            log.addHandler(file_handler)
        return os.path.join(log_dir, logger_path)

    elif mode == 'continue':
        assert config.run.model_path is not None
        if rank == 0:
            file_path = os.path.join(log_dir, config.run.model_path, 'info.log')
            file_handler = logging.FileHandler(file_path, mode='a')
            file_handler.setFormatter(log_format)
            log.addHandler(file_handler)
        return os.path.join(log_dir, config.run.model_path)

    elif mode == 'test':
        return "log/tmp/"
    else:
        raise NotImplementedError


def get_writer(folder, flush_secs=120):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    writer = SummaryWriter(folder, flush_secs=flush_secs)
    return writer


def get_bert_config(bert_path):
    bert_config_path = os.path.join(bert_path, "config.json")
    bert_config = json.load(open(bert_config_path))

    return EasyDict(bert_config)


model_dict = {
    'JointModel': JointModel,
    'LayoutLM': LayoutLM,
    'Baseline': Baseline,
}

if __name__ == '__main__':

    rank, world_size = initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='config/default.yaml', type=str)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf8') as fc:
        config = EasyDict(yaml.load(fc))

    set_random_seed(config.run.random_seed)
    root_path = set_logger(config.run.mode)
    if config.run.mode in ['train', 'continue'] and rank == 0:
        writer = get_writer(folder=os.path.join(root_path, "tensorboard"))
    else:
        writer = None
    config.tensorboard = writer

    bert_config = get_bert_config(bert_path=config.model.nlp.bert_weight)
    config.model.nlp.bert_config = bert_config

    logging.info('Running begin!')

    assert config.model.name in model_dict
    model = model_dict[config.model.name](config).to(torch.device(config.run.device))
    model = DistModule(model, config.linklink.sync)

    loss_fn = getattr(loss, config.loss.func)(config)
    eval_func = getattr(eval, config.eval.func)(config)

    logging.info('Running mode: {}'.format(config.run.mode))

    if config.run.mode in ['train', 'continue', 'test']:
        train_loader, train_val_loader, val_loader = load_data(config)
        train(model, loss_fn, eval_func, train_loader, train_val_loader, val_loader, config, root_path)

    if writer is not None:
        writer.close()

    link.finalize()
