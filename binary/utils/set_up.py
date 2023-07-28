import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
import os
import pickle as pkl

def save_dataset(epochs, train_labels, label_map, ratio, model_dir):
    ds_list = Dataset.get_data_splits_list(epochs, train_labels, label_map, ratio)
    data_root = os.path.join('data', model_dir[:2])
    Config.check_dir(data_root)
    for idx, ds in enumerate(ds_list):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'wb') as f:
            pkl.dump( ds, f)
        print(data_path, 'saved')

def load_dataset(epochs, ratio, model_dir):
    data_root = os.path.join('data', model_dir[:2])
    ds_list = []
    for idx in range(epochs):
        data_path = os.path.join(data_root, '{}.pt'.format(idx))
        with open(data_path, 'rb') as f:
            ds_dict = pkl.load(f) 
        print(data_path, 'loaded')

        final_dict = Dataset.load_dataset(ds_dict, ratio['remove_rate'], model_dir)    
        ds_list.append(final_dict)
    return ds_list

def set_up(epochs, model_dir, device_id=0):
    batch_size, train_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse()
    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)
    ds_list = load_dataset(epochs, ratio, model_dir)
    return batch_size, new_img_num_list, superclass_num, seq_rounds_config, device_config, ds_list
