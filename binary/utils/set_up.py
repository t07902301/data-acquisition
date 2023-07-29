import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
import os
import pickle as pkl
from utils.env import data_split_env

def save_dataset(epochs, train_labels, label_map, ratio, root_model_dir):
    data_split_env()
    ds_list = Dataset.get_data_splits_list(epochs, train_labels, label_map, ratio)
    data_root = os.path.join('data', root_model_dir)
    Config.check_dir(data_root)
    for idx, ds in enumerate(ds_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
        print(data_path, 'saved')

def load_dataset(epochs, ratio, model_dir):
    data_root = os.path.join('data', model_dir[:2])
    ds_list = []
    remove_labels = Dataset.data_config['remove_fine_labels'][model_dir]
    print('Removed Labels:', remove_labels)
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_dataset(ds_dict, ratio['remove_rate'], remove_labels)    
        ds_list.append(final_dict)
    return ds_list

def load_cover_dataset(epochs, ratio, model_dir):
    data_root = os.path.join('data', model_dir[:2])
    ds_list = []
    cover_labels = Dataset.data_config['cover_labels'][model_dir]
    print('Covered Labels:', cover_labels)
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_cover_dataset(ds_dict, ratio['remove_rate'], cover_labels)    
        ds_list.append(final_dict)
    return ds_list

def set_up(epochs, model_dir, device_id=0):
    data_split_env()
    batch_size, train_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse()
    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)
    ds_list = load_dataset(epochs, ratio, model_dir)
    # ds_list = load_cover_dataset(epochs, ratio, model_dir)
    return batch_size, new_img_num_list, superclass_num, seq_rounds_config, device_config, ds_list
