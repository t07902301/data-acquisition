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

    indices_list, normalize_stat = dataset.create(epochs, config, meta_path)

    data_root = os.path.join('data', model_dir[:2])
    Config.check_dir(data_root)
    for idx, ds in enumerate(indices_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
            f.close()
        logger.info('save dataset indices to {}'.format(data_path))
    
    # save_stat(normalize_stat, data_root)

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
    data_root = os.path.join('data', model_dir)
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def load_dataset(epochs, data_dir, data_config, option, dataset:dataset_utils.Dataset, normalized_stat):
    data_root = os.path.join('data', data_dir)
    ds_list = []
    remove_rate = data_config['ratio']['remove']

    # meta_path = os.path.join('data/meta/s2.pkl')
    
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            dataset_raw_indices = pkl.load(f) 
        logger.info('dataset loaded from {}'.format(data_path))
        # dataset_dict = dataset.load_dataset_raw_indices(dataset_raw_indices, data_config, normalized_stat, meta_path)
        final_dict = dataset.load(dataset_raw_indices, data_config, normalized_stat)
        # (dataset_raw_indices, remove_rate, data_config['labels'], option)    
        ds_list.append(final_dict)
    return ds_list

def load_cover_dataset(epochs, model_dir, data_config, dataset:dataset_utils.Cifar):
    data_root = os.path.join('data', model_dir)
    ds_list = []
    remove_rate = data_config['ratio']['remove_rate']
    cover_labels = data_config['labels']['cover']
    logger.info('Covered Labels:{}'.format(cover_labels))

    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        logger.info('dataset loaded from {}'.format(data_path))
        final_dict = dataset.load_cover(ds_dict, remove_rate, cover_labels)    

        ds_list.append(final_dict)
    return ds_list

import yaml

def load_config(model_dir):
    config_path = os.path.join('log', model_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        file.close()
    return config

def set_up(epochs, model_dir, device_id, option, dataset_name):


    data_split_env()

    data_dir = model_dir[:2] # For shift degree control

    # normalize_stat = load_stat(data_dir)
    normalize_stat = {'mean': [0.4794902205467224, 0.4533461928367615, 0.387584924697876], 'std': [0.2391287088394165, 0.2298320084810257, 0.2330065220594406]}


    config = load_config(model_dir)

    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)

    label_map = config['data']['labels']['map']
    logger.info('Label Map:{}'.format(label_map))
    
    dataset = dataset_utils.factory(dataset_name)

    ds_list = load_dataset(epochs, data_dir, config['data'], option, dataset, normalize_stat)

    return config, device_config, ds_list, normalize_stat

