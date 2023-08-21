import torchvision.transforms as transforms
import torch
import numpy as np
from utils.env import generator
import utils.dataset.core as core
import utils.objects.Config as Config
from utils.meta_data import *

n_workers = 1

class DataSplits():
    dataset: dict
    loader: dict
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        generator.manual_seed(0)    
        self.loader = {
            k: torch.utils.data.DataLoader(self.dataset[k], batch_size=batch_size, 
                                        shuffle=(k=='train'), drop_last=(k=='train'),
                                        num_workers=n_workers,generator=generator)
            for k in self.dataset.keys()
        }        
        self.batch_size = batch_size

    def expand(self, split_name, new_data):
        '''
        dataset (dict) is mutable, pass by reference
        '''
        self.dataset[split_name] = torch.utils.data.ConcatDataset([self.dataset[split_name],new_data])
        self.update_dataloader(split_name) 

    def reduce(self, split_name, remove_data_indices):
        '''
        dataset (dict) is mutable, pass by reference
        '''
        original_split_len = len(self.dataset[split_name])
        left_market_mask = complimentary_mask(mask_length=original_split_len,active_spot=remove_data_indices)
        self.dataset[split_name] = torch.utils.data.Subset(self.dataset[split_name],np.arange(original_split_len)[left_market_mask])
        self.update_dataloader(split_name) 

    def update_dataloader(self, split_name):
        generator.manual_seed(0)        
        self.loader[split_name] = torch.utils.data.DataLoader(self.dataset[split_name], batch_size= self.batch_size, 
                                                              shuffle=(split_name=='train'), drop_last=(split_name=='train'),
                                                              num_workers=n_workers, generator=generator)
        
    def replace(self, replaced_name, new_data):
        self.dataset[replaced_name] = new_data
        self.update_dataloader(replaced_name)

    def use_new_data(self, new_data, new_model_config:Config.NewModel, acquisition_config:Config.Acquisition, target_name:str):
        '''
        new data to be added to train set or not, and update loader automatically
        '''
        assert len(new_data) == acquisition_config.n_ndata, 'size error - new data: {}, required new data: {} \n under {}'.format(len(new_data), acquisition_config.n_ndata, acquisition_config.get_info())

        if new_model_config.pure:
            self.replace(target_name, new_data)
        else:
            self.expand(target_name, new_data)

def primal_operate(ds_dict, remove_rate, remove_label_config, option):

    new_labels_remove_rate = remove_rate['new_labels']
    old_labels_remove_rate = remove_rate['old_labels']

    subclass = subclass_config(remove_label_config, option)
    print(option, 'Removed Labels:', subclass['new'])

    _, old_aug_train = split_dataset(ds_dict['aug_train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_train = split_dataset(ds_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_val = split_dataset(ds_dict['val'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_test = split_dataset(ds_dict['test'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    
    if old_labels_remove_rate != None:
        _, val = split_dataset(ds_dict['val'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
        _, test = split_dataset(ds_dict['test'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
    else:
        val, test = ds_dict['val'], ds_dict['test']

    # sub_mkt, _ = split_dataset(ds_dict['market'], subclass['new'] + subclass['old'], 0.1)

    return {
        'train': old_aug_train,
        'train_non_cnn': old_train,
        'val': old_val,
        'test': old_test,
        'val_shift': val,
        'test_shift': test,
        'market': ds_dict['market'],
        'aug_market': ds_dict['aug_market'],
        # 'sub_mkt': sub_mkt
    }

def dual_operate(primal_dict, remove_rate, remove_label_config, option):

    new_labels_remove_rate = remove_rate['new_labels']
    old_labels_remove_rate = remove_rate['old_labels']
    
    subclass = subclass_config(remove_label_config, option)
    print(option, 'Removed Labels:', subclass['new'])

    _, old_aug_train = split_dataset(primal_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_train = split_dataset(primal_dict['train_non_cnn'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_val = split_dataset(primal_dict['val'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    _, old_test = split_dataset(primal_dict['test'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
    
    if old_labels_remove_rate != None:
        _, val = split_dataset(primal_dict['val_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
        _, test = split_dataset(primal_dict['test_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
    else:
        val, test = primal_dict['val_shift'], primal_dict['test_shift']
        
    return {
        'train': old_aug_train,
        'train_non_cnn': old_train,
        'val': old_val,
        'test': old_test,
        'val_shift': val,
        'test_shift': test,
        'market': primal_dict['market'],
        'aug_market': primal_dict['aug_market'],
    }

def load_dataset(ds_dict, remove_rate, remove_label_config, option):
    if option == 'both':
        primal_dict = primal_operate(ds_dict, remove_rate, remove_label_config, option = 'session')
        return dual_operate(primal_dict, remove_rate, remove_label_config, option='object')
    else:
        return primal_operate(ds_dict, remove_rate, remove_label_config, option)

def get_split_indices(dataset_labels, target_labels, split_1_ratio):
    '''
    Label-wise split a dataset into two parts according to target labels and a ratio. 
    
    Params: 
           split_1_ratio: how much from target labels assigned to PART 1
    Return:
           p1_indices: indices belonging to part 1 (split_1_ratio)
           p2_indices: indices belonging to part 2
    '''

    if torch.is_tensor(dataset_labels):
        dataset_labels = dataset_labels.numpy()
    ds_length = len(dataset_labels)

    if len(target_labels) == 0 or split_1_ratio == 0:
        return None, np.arange(ds_length)
    
    p1_indices = []

    for c in target_labels:
        cls_indices = np.arange(ds_length)[dataset_labels == c]
        split_indices = sample_indices(cls_indices,split_1_ratio)
        p1_indices.append(split_indices)

    p1_indices = np.concatenate(p1_indices)
    mask_p2 = complimentary_mask(mask_length=ds_length,active_spot=p1_indices)
    p2_indices = np.arange(ds_length)[mask_p2]
    return p1_indices,p2_indices

def split_dataset(dataset, target_ratio, option: dict):
    '''
    Split Dataset by Target Labels \n
    Return (target dataset, the rest)
    '''

    option_name = option['name']
    option_labels = option['labels']
    dataset_labels = get_labels(dataset, option_name)

    target_splits, other_splits = get_split_indices(dataset_labels, option_labels, target_ratio)

    target_subset = torch.utils.data.Subset(dataset, target_splits)
    other_subset = torch.utils.data.Subset(dataset, other_splits)

    return target_subset, other_subset

def choose_label_index(option):
    if option == 'object':
        return 2
    elif option == 'session':
        return 3
    else:
        return 1
    
def get_labels(ds, option:str):
    ds_size = len(ds)
    label_idx = choose_label_index(option)
    labels = []
    for idx in range(ds_size):
        label = ds[idx][label_idx]
        labels.append(label)
    return np.array(labels)
    
def sample_indices(indices,ratio):
    if type(ratio) == int and ratio!=1:
        return np.random.choice(indices,ratio,replace=False)
    else:
        return np.random.choice(indices,int(ratio*len(indices)),replace=False)

def create_dataset(meta_split:dict, mean, std, label_map):
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
    
    augment_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        base_transform
    ])

    return {
        'val': core.Core(meta_split['val'], base_transform, label_map),
        'test': core.Core(meta_split['test'], base_transform, label_map),
        'train': core.Core(meta_split['train'], base_transform, label_map),
        'market': core.Core(meta_split['market'], base_transform, label_map),
        'aug_train': core.Core(meta_split['train'], augment_transform, label_map),
        'aug_market': core.Core(meta_split['market'], augment_transform, label_map),
    }

def subclass_config(remove_config, option = 'session'):
    if option == 'session':
        sessions = [i for i in range(11)]
        left_sessions = set(sessions) - set(remove_config['session'])
        return {
            'new': remove_config['session'],
            'old': left_sessions
        }
    else:
        objects = [i for i in range(50)]
        left_objects = set(objects) - set(remove_config['object'])
        return {
            'new': remove_config['object'],
            'old': left_objects
        }
    
def get_data_splits_list(epochs, config, meta_path):
    
    data_config = config['data']
    label_map = data_config['label_map']

    fo = open(meta_path, 'rb')
    meta_data = pkl.load(fo)    
    fo.close()

    mean, std = normalize(meta_data)

    normalize_stat = {
        'mean': mean,
        'std': std
    }

    ds_list = []
    for epo in range(epochs):
        meta_split = get_meta_splits(data_config, meta_data)
        ds = create_dataset(meta_split, mean, std, label_map)
        ds_list.append(ds)
    return ds_list, normalize_stat

def normalize(meta_data):
    train_ds = core.Core(meta_data)
    r, g, b = [], [], []
    train_size = len(train_ds)
    for idx in range(train_size):
        raw_img = train_ds[idx][0]
        img = transforms.ToTensor()(raw_img)
        r.append(img[0])
        g.append(img[1])
        b.append(img[2])
    result = [r, g, b]
    mean = []
    std = []
    for color in result:
        color = torch.concat(color) # list to tensor
        mean.append(torch.mean(color).item())
        std.append(torch.std(color).item())
    return mean, std #superclass

def get_indices_by_labels(ds, subset_labels, option):
    select_indice = []
    ds_labels = get_labels(ds, option)
    for c in subset_labels:
        select_indice.append(np.arange(len(ds))[c==ds_labels])
    select_indice = np.concatenate(select_indice)
    return select_indice

def get_subset_by_labels(ds, subset_labels, option, return_split=False):
    select_indice = get_indices_by_labels(ds, subset_labels, option)
    ds = torch.utils.data.Subset(ds,select_indice) 
    if return_split:
        return ds, select_indice
    else:
        return ds, None

def modify_coarse_label(dataset, label_map):
    for split in dataset.keys():
        dataset[split] = core.ModifiedDataset(dataset=dataset[split], category_transform=label_map)
    return dataset

def get_vis_transform(mean, std):
    # For visualization
    INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/x for x in std]),
                                transforms.Normalize(mean = [-x for x in mean],
                                                     std = [ 1., 1., 1. ]),])
    TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])
    return INV_NORM, TOIMAGE

