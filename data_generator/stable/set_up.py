import utils.objects.Config as Config
import utils.dataset.wrappers as dataset_utils
import torch
import os
import pickle as pkl
from utils.env import data_split_env
import numpy as np
from utils.logging import *

def save_dataset(epochs, config, model_dir, dataset_name):

    data_split_env()

    dataset = dataset_utils.factory(dataset_name)

    meta_dir = model_dir[:2] # How to split meta data, None for Cifar
    meta_path = os.path.join('data/meta', '{}.pkl'.format(meta_dir))

    indices_list, normalize_stat = dataset.get_dataset_raw_indices(epochs, config, meta_path)

    data_root = os.path.join('data', model_dir)
    Config.check_dir(data_root)
    for idx, ds in enumerate(indices_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
            f.close()
        logger.info('save dataset indices to {}'.format(data_path))
    save_stat(normalize_stat, data_root)

    split_data = ds
    for spit_name in split_data.keys():
        logger.info('{}: {}'.format(spit_name, len(split_data[spit_name])))

def save_stat(stat_dict, data_root):
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'wb') as f:
        pkl.dump(stat_dict, f)
        f.close()
    logger.info('stat saved to {}'.format(stat_path))

def load_stat(model_dir):
    data_root = os.path.join('init_data', model_dir)
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def load_dataset(epochs, data_dir, data_config, option, dataset:dataset_utils.Dataset, normalized_stat, model_dir):
    data_root = os.path.join('init_data', data_dir)
    ds_list = []
    remove_rate = data_config['ratio']['remove']

    meta_path = os.path.join('init_data/meta/s2.pkl')
   
    sub_data_root = os.path.join('data', model_dir)
    
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('dataset loaded from {}'.format(data_path))
        dataset_dict = dataset.load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat, meta_path)
        final_dict, final_raw_indices = dataset.load(dataset_dict, remove_rate, data_config['labels'], option)    
        ds_list.append(final_dict)

        sub_data_path = os.path.join(sub_data_root, '{}.pt'.format(idx))
        with open(sub_data_path, 'wb') as f:
            pkl.dump(final_raw_indices, f)
            f.close()
        logger.info('save sub dataset indices to {}'.format(sub_data_path))

    return ds_list

def sub_load_dataset(epochs, data_dir, data_config, option, dataset:dataset_utils.Dataset, normalized_stat):
    data_root = os.path.join('data', data_dir)
    ds_list = []

    meta_path = os.path.join('init_data/meta/s2.pkl')
   
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('dataset loaded from {}'.format(data_path))
        dataset_dict = dataset.sub_load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat, meta_path)
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

    normalize_stat = load_stat('{}_{}'.format(dataset_name, task))

    config = load_config(model_dir)

    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)

    label_map = config['data']['labels']['map']
    logger.info('Label Map:{}'.format(label_map))
    
    dataset = dataset_utils.factory(dataset_name)

    ds_list = load_dataset(epochs, '{}_{}'.format(dataset_name, task), config['data'], task, dataset, normalize_stat, model_dir)

    return config, device_config, ds_list, normalize_stat, dataset_name, task

