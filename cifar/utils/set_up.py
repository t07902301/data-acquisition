import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
import os
import pickle as pkl
from utils.env import data_split_env

def save_dataset(epochs, config, model_dir):
    data_split_env()
    ds_list, normalize_stat = Dataset.get_data_splits_list(epochs, config)
    data_root = os.path.join('data', model_dir)
    Config.check_dir(data_root)
    for idx, ds in enumerate(ds_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
            f.close()
        print(data_path, 'saved')
    save_stat(normalize_stat, data_root)

    split_data = ds
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))

def save_stat(stat_dict, data_root):
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'wb') as f:
        pkl.dump(stat_dict, f)
        f.close()
    print(stat_path, 'saved')

def load_stat(model_dir):
    data_root = os.path.join('data', model_dir)
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def load_dataset(epochs, model_dir, data_config):
    data_root = os.path.join('data', model_dir)
    ds_list = []
    remove_labels = data_config['remove_fine_labels']
    remove_rate = data_config['ratio']['remove_rate']
    print('Removed Labels:', remove_labels)
    old_labels = list(set(data_config['select_fine_labels']) - set(remove_labels))

    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_dataset(ds_dict, remove_rate, remove_labels, old_labels)    
        ds_list.append(final_dict)
    return ds_list

def load_cover_dataset(epochs, model_dir, data_config):
    data_root = os.path.join('data', model_dir)
    ds_list = []
    remove_rate = data_config['ratio']['remove_rate']
    cover_labels = data_config['cover_labels']
    print('Covered Labels:', cover_labels)

    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_cover_dataset(ds_dict, remove_rate, cover_labels)    

        ds_list.append(final_dict)
    return ds_list

import yaml

def load_config(model_dir):
    config_path = os.path.join('log', model_dir, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        file.close()

    return config

def set_up(epochs, model_dir, device_id=0):

    data_split_env()

    data_dir = model_dir[:3] # For shift degree control

    normalize_stat = load_stat(data_dir)

    config = load_config(model_dir)

    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)

    select_fine_labels = config['data']['select_fine_labels']
    label_map = config['data']['label_map']
    print('Label Map:', label_map)
    print('select_fine_labels:', select_fine_labels)

    # ds_list = load_dataset(epochs, data_dir, config['data'])
    ds_list = load_cover_dataset(epochs, data_dir, config['data'])

    return config, device_config, ds_list, normalize_stat
