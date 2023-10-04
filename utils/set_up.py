import utils.objects.Config as Config
import utils.dataset.wrappers as dataset_utils
import torch
import os
import pickle as pkl
from utils.env import data_split_env
import numpy as np
from utils.logging import *

def load_stat(model_dir):
    data_root = os.path.join('data', model_dir)
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def load_dataset(epochs, model_dir, data_config, dataset:dataset_utils.Dataset, normalized_stat):
    ds_list = []
    for idx in range(epochs):
        data_path = os.path.join('data', model_dir, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('dataset loaded from {}'.format(data_path))
        dataset_dict = dataset.load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat)
        ds_list.append(dataset_dict)
    return ds_list

import yaml

def load_config(model_dir):
    config_path = os.path.join('log', model_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        file.close()
    return config

def parse(filename:str):
    parse_list = filename.split('_')
    if len(parse_list) == 2:
        dataset_name, task = parse_list
        aux = None
    else:
        dataset_name, task, aux = parse_list
    return dataset_name, task, aux

def set_up(epochs, model_dir, device_id):

    data_split_env()

    dataset_name, task, aux = parse(model_dir)

    normalize_stat = load_stat(model_dir)

    config = load_config(model_dir)

    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)

    label_map = config['data']['labels']['map']
    logger.info('Label Map:{}'.format(label_map))
    
    dataset = dataset_utils.factory(dataset_name)

    ds_list = load_dataset(epochs, model_dir, config['data'], dataset, normalize_stat)

    return config, device_config, ds_list, normalize_stat, dataset_name, task

