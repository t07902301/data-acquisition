import utils.objects.Config as Config
import utils.dataset.wrappers as dataset_utils
# import data_generator.stable.wrappers as dataset_utils

import torch
import os
import pickle as pkl
from utils.env import data_env
import numpy as np
from utils.logging import *

def save_dataset_split(epochs, model_dir, config):
    '''
    Generate indices of train, test. validation, and data pool
    '''
    data_env()

    dataset_name, task, aux = parse(model_dir)
    dataset = dataset_utils.factory(dataset_name)

    indices_list, normalize_stat = dataset.create_4_splits(epochs, config)
    dataset_name, task, aux = parse(model_dir)

    split_dir = 'init_data/{}_{}'.format(dataset_name, task)
    Config.check_dir(split_dir)
    for idx, ds in enumerate(indices_list):
        data_path = os.path.join(split_dir, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
            f.close()
        logger.info('save data splits indices to {}'.format(data_path))
   
    save_stat(normalize_stat, split_dir)

    split_data = ds
    for spit_name in split_data.keys():
        logger.info('{}: {}'.format(spit_name, len(split_data[spit_name])))

def save_stat(stat_dict, root):
    '''
    Generate mean and std for normalizing data
    '''
    stat_path = os.path.join(root, 'normalize_stat.pt')
    with open(stat_path, 'wb') as f:
        pkl.dump(stat_dict, f)
        f.close()
    logger.info('normalize stat saved to {}'.format(stat_path))

def load_stat(root):
    stat_path = os.path.join(root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def save_dataset_shift(epochs, model_dir, config):
    data = []
    data_config = config['data']
    remove_rate = data_config['ratio']['remove']
    dataset_name, task, aux = parse(model_dir)
    dataset = dataset_utils.factory(dataset_name)

    split_dir = 'init_data/{}_{}'.format(dataset_name, task)
    shift_dir = 'data/{}'.format(model_dir)
    normalized_stat = load_stat(split_dir)

    for idx in range(epochs):
        split_path = os.path.join(split_dir, '{}.pt'.format(idx))
        with open(split_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('data split loaded from {}'.format(split_path))
        split_dict = dataset.load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat)
        _, shift_raw_indices = dataset.create_shift(split_dict, remove_rate, data_config['labels'], task)    
        data.append(shift_raw_indices)

        shift_path = os.path.join(shift_dir, '{}.pt'.format(idx))
        with open(shift_path, 'wb') as f:
            pkl.dump(shift_raw_indices, f)
            f.close()
        logger.info('save data shift indices to {}'.format(shift_path))

    return data

def load_dataset(epochs, root, data_config, dataset:dataset_utils.Dataset, normalized_stat):
    data = []

    for idx in range(epochs):
        data_path = os.path.join(root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('data loaded from {}'.format(data_path))
        data_dict = dataset.load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat)
        data.append(data_dict)
    return data

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
    '''
    Load data shifts indices, normalize stat and config
    '''

    data_env()

    dataset_name, task, aux = parse(model_dir)

    split_dir = 'init_data/{}_{}'.format(dataset_name, task)

    normalize_stat = load_stat(split_dir)

    config = load_config(model_dir)

    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)

    dataset = dataset_utils.factory(dataset_name)

    data = load_dataset(epochs, split_dir, config['data'], dataset, normalize_stat)

    return config, device_config, data, normalize_stat, dataset_name, task

