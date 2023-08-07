import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
import os
import pickle as pkl
from utils.env import data_split_env

def save_dataset(epochs, select_fine_labels, label_map, ratio, task_id):
    data_split_env()
    ds_list, normalize_stat = Dataset.get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    data_root = os.path.join('data', task_id)
    Config.check_dir(data_root)
    for idx, ds in enumerate(ds_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
            f.close()
        print(data_path, 'saved')
    save_stat(normalize_stat, data_root)

def save_stat(stat_dict, data_root):
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'wb') as f:
        pkl.dump(stat_dict, f)
        f.close()
    print(stat_path, 'saved')

def load_stat(model_dir):
    task_id = model_dir[:2]
    data_root = os.path.join('data', task_id)
    stat_path = os.path.join(data_root, 'normalize_stat.pt')
    with open(stat_path, 'rb') as f:
        normalize_stat = pkl.load(f) 
        f.close()
    return normalize_stat

def load_dataset(epochs, remove_rate, model_dir, select_fine_labels):
    task_id, sub_task_id = model_dir[:2], model_dir[:3]
    data_root = os.path.join('data', task_id)
    ds_list = []
    remove_labels = Dataset.data_config['remove_fine_labels'][sub_task_id]
    print('Removed Labels:', remove_labels)
    old_labels = list(set(select_fine_labels) - set(remove_labels))
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_dataset(ds_dict, remove_rate, remove_labels, old_labels)    
        ds_list.append(final_dict)
    return ds_list

def load_cover_dataset(epochs, remove_rate, model_dir, select_fine_labels):
    task_id, sub_task_id = model_dir[:2], model_dir[:3]
    data_root = os.path.join('data', task_id)
    ds_list = []
    cover_labels = Dataset.data_config['cover_labels'][sub_task_id]
    print('Covered Labels:', cover_labels)
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')
        final_dict = Dataset.load_cover_dataset(ds_dict, remove_rate, cover_labels, select_fine_labels)    
        ds_list.append(final_dict)
    return ds_list

def set_up(epochs, model_dir, device_id=0):
    data_split_env()
    normalize_stat = load_stat(model_dir)
    batch_size, total_labels, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse()
    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)
    task_id = model_dir[:2]
    select_fine_labels = total_labels[task_id]['select_fine_labels']
    label_map = total_labels[task_id]['label_map']
    print('Label Map:', label_map)
    print('select_fine_labels:', select_fine_labels)
    # ds_list = load_dataset(epochs, ratio['remove_rate'], model_dir, select_fine_labels)
    ds_list = load_cover_dataset(epochs, ratio['remove_rate'], model_dir, select_fine_labels)

    return batch_size, new_img_num_list, superclass_num, seq_rounds_config, device_config, ds_list, normalize_stat
