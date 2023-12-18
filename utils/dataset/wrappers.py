import torchvision.transforms as transforms
import torch
from utils.env import dataloader_env
import utils.objects.Config as Config
from abc import abstractmethod
import utils.dataset.cifar as cifar
import utils.dataset.core as core
import pickle as pkl
import numpy as np
from utils import n_workers
from utils.logging import *
from typing import Dict

def complimentary_mask(mask_length, active_spot):
    '''
    reverse the active mask
    '''
    active_mask = np.zeros(mask_length, dtype=bool)
    active_mask[active_spot] = True
    reversed_mask = ~active_mask 
    return reversed_mask

def get_vis_transform(mean, std):
    # For visualization
    INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/x for x in std]),
                                transforms.Normalize(mean = [-x for x in mean],
                                                    std = [ 1., 1., 1. ]),])
    TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])
    return TOIMAGE

class MetaData():
    def __init__(self, path) -> None:
        data = self.load(path)
        self.object_labels = np.asarray(data['object'])
        self.session_labels = np.asarray(data['session'])
        self.category_labels = np.asarray(data['category'])
        self.data = np.asarray(data['data'])

    def load(self, path):
        fo = open(path, 'rb')
        meta_data = pkl.load(fo)    
        fo.close()
        return meta_data

    def get_subset(self, dataset_labels, targets, dataset_indices, ratio=0):
        '''
        Find indices of subsets where labels match with targets
        '''
        
        results = []

        for target in targets:

            subset_indicies = dataset_indices[dataset_labels==target]

            if ratio > 0 :
                if type(ratio) == float:
                    subset_indicies = np.random.choice(subset_indicies, int(ratio * len(subset_indicies)), replace=False)
                else:
                    subset_indicies = np.random.choice(subset_indicies, ratio, replace=False)

            results.append(subset_indicies)
            
        results = np.concatenate(results)

        return results

    def balanced_split(self, ratio, categories, sessions, range_indices):
        '''
        Split Dataset fairly all the way from category, session to object. Ratio for sampling under object levels.\n
        Return relative indices to the input range
        '''
        category_labels = self.category_labels[range_indices]
        session_labels = self.session_labels[range_indices]
        object_labels = self.object_labels[range_indices]

        dataset_size = len(range_indices)
        
        relative_indices = np.arange(dataset_size)

        sampled_cat_indices = []

        for cat in categories:

            cat_indices = relative_indices[category_labels==cat]
            cat_session_labels = session_labels[cat_indices]

            objects = [i + cat * 5 for i in range(5)]

            sampled_session_indices = []
            for session in sessions: # session labels given a cat

                cat_session_indices = cat_indices[cat_session_labels==session]
                cat_session_obj_labels = object_labels[cat_session_indices]

                subset_indices = self.get_subset(cat_session_obj_labels, objects, cat_session_indices, ratio) # obj labels given a session, frames sampled
                sampled_session_indices.append(subset_indices)
            
            sampled_session_indices = np.concatenate(sampled_session_indices, axis = 0)

            sampled_cat_indices.append(sampled_session_indices)

        sampled_cat_indices = np.concatenate(sampled_cat_indices, axis = 0)

        return {
            'sampled': sampled_cat_indices,
            'others': relative_indices[complimentary_mask(dataset_size, sampled_cat_indices)]
        }

    def subset2dict(self, subset_indices, range_indices):
        data = self.data[range_indices]
        category_labels = self.category_labels[range_indices]
        session_labels = self.session_labels[range_indices]
        object_labels = self.object_labels[range_indices]
        return {
            'data': data[subset_indices],
            'session': session_labels[subset_indices],
            'object': object_labels[subset_indices],
            'category': category_labels[subset_indices]
        }

class DataSplits():
    dataset: dict
    loader: dict
    def __init__(self, dataset, batch_size, name=None) -> None:
        self.dataset = dataset
        self.dataset_name = name
        generator = dataloader_env()
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
        generator = dataloader_env()
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
        # assert len(new_data) == acquisition_config.n_ndata, 'size error - new data: {}, required new data: {} \n under {}'.format(len(new_data), acquisition_config.n_ndata, acquisition_config.get_info())
        logger.info('Acquired Data: {}'.format(len(new_data)))

        if new_model_config.pure:
            self.replace(target_name, new_data)
        else:
            self.expand(target_name, new_data)

class Dataset():
    '''
    Dataset Interface
    '''
    def __init__(self) -> None:
        pass

    def get_raw_indices(self, dataset):
        '''
        Reture indices to the original dataset (Train & Test in Cifar, One dataset in Core)
        '''
        raw_indices = [dataset[idx][-1] for idx in range(len(dataset))]
        return raw_indices
    
    @abstractmethod
    def load_dataset_raw_indices(self, raw_indices_dict:Dict[str, list], data_config, normalized_stat):
        '''
        Load indices of data shift or split given the number of keys in the indices dict.
        '''
        pass
    
    @abstractmethod
    def get_labels(self):
        '''
        Iterate over dataset to get labels
        '''
        pass

    @abstractmethod
    def normalize(self):
        '''
        Iterate over dataset to get RGB stat
        '''
        pass

    @abstractmethod
    def split(self):
        '''
        Split dataset by given labels and ratio (int or float)
        '''
        pass

    @abstractmethod
    def create_shift(self, ds_dict, remove_rate, label_config, option):
        '''
        Get dataset shifts indices from loading shifted data splits
        '''
        pass

    @abstractmethod
    def create(self):
        '''
        Create dataset splits by ratios from config
        '''
        pass

    @abstractmethod
    def create_4_splits(self, epochs, config):
        '''
        Get dataset splits from raw or meta dataset
        '''
        pass

    @abstractmethod
    def get_indices_by_labels(self, ds, subset_labels):
        pass

    @abstractmethod
    def get_subset_by_labels(self, ds, subset_labels, return_split=False):
        pass

    def sample_indices(self, indices, ratio):
        if type(ratio) == int and ratio!=1:
            return np.random.choice(indices,ratio,replace=False)
        else:
            return np.random.choice(indices,int(ratio*len(indices)),replace=False)
        
    def get_split_indices(self, dataset_labels, target_labels, split_1_ratio):
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
        dataset_indices = np.arange(ds_length)

        if len(target_labels) == 0 or split_1_ratio == 0:
            return None, dataset_indices
        
        p1_indices = []


        for c in target_labels:
            cls_indices = dataset_indices[dataset_labels == c]
            split_indices = self.sample_indices(cls_indices,split_1_ratio)
            p1_indices.append(split_indices)

        p1_indices = np.concatenate(p1_indices)
        mask_p2 = complimentary_mask(mask_length=ds_length,active_spot=p1_indices)
        p2_indices = dataset_indices[mask_p2]
        return p1_indices,p2_indices

class Cifar(Dataset):
    
    def __init__(self) -> None:
        super().__init__()

    def load_dataset_raw_indices(self, raw_indices_dict:Dict[str, list], data_config, normalized_stat):
        raw_ds = self.get_raw_dataset(data_config['root'], normalized_stat, data_config['labels']['map'])
        raw_dataset_split = {
            'val_shift': torch.utils.data.Subset(raw_ds['test_val'], raw_indices_dict['val_shift']),
            'test_shift': torch.utils.data.Subset(raw_ds['test_val'], raw_indices_dict['test_shift']),
            'train': torch.utils.data.Subset(raw_ds['train_market'], raw_indices_dict['train']),
            'market': torch.utils.data.Subset(raw_ds['train_market'], raw_indices_dict['market']),
            # 'aug_train': torch.utils.data.Subset(raw_ds['aug_train_market'], raw_indices_dict['train']),
            # 'aug_market': torch.utils.data.Subset(raw_ds['aug_train_market'], raw_indices_dict['market']),
        }
        if len(list(raw_indices_dict.keys()))>4:
            raw_dataset_split['train_non_cnn'] = torch.utils.data.Subset(raw_ds['train_market'], raw_indices_dict['train'])
            raw_dataset_split['val'] = torch.utils.data.Subset(raw_ds['test_val'], raw_indices_dict['val'])
            raw_dataset_split['test'] = torch.utils.data.Subset(raw_ds['test_val'], raw_indices_dict['test'])   
        return raw_dataset_split

    def get_indices_by_labels(self, ds, subset_labels, use_fine_label):
        select_indice = []
        ds_labels = self.get_labels(ds, use_fine_label)
        for c in subset_labels:
            select_indice.append(np.arange(len(ds))[c==ds_labels])
        select_indice = np.concatenate(select_indice)
        return select_indice

    def get_subset_by_labels(self, ds, subset_labels, return_indices=False, use_fine_label=True):
        select_indice = self.get_indices_by_labels(ds, subset_labels, use_fine_label)
        ds = torch.utils.data.Subset(ds,select_indice) 
        if return_indices:
            return ds, select_indice
        else:
            return ds, None
        
    def create_shift(self, ds_dict, remove_rate, label_config, option):
        if 'cover' in label_config:
            return self.cover_load(ds_dict, remove_rate, label_config['cover'])
        else:
            return self.non_cover_load(ds_dict, remove_rate, label_config)

    def non_cover_load(self, ds_dict:Dict[str, cifar.CIFAR100], remove_rate, label_config):
        new_labels = label_config['remove']
        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']
        total_labels = label_config['select_fine_labels'] if len(label_config['select_fine_labels']) != 0 else [i for i in range(100)]
        old_labels = list(set(total_labels) - set(new_labels)) 
        logger.info('Fine Labels removed: {}'.format(new_labels))

        # _, old_aug_train = self.split(ds_dict['aug_train'], new_labels, new_labels_remove_rate)
        _, old_train, split_indices = self.split(ds_dict['train'], new_labels, new_labels_remove_rate, return_split_indices=True)
        # old_aug_train = torch.utils.data.Subset(ds_dict['aug_train'], split_indices['other'])
        _, old_val = self.split(ds_dict['val_shift'], new_labels, new_labels_remove_rate)
        _, old_test = self.split(ds_dict['test_shift'], new_labels, new_labels_remove_rate)
        
        if old_labels_remove_rate != None:
            _, val = self.split(ds_dict['val_shift'], old_labels, old_labels_remove_rate)
            _, test = self.split(ds_dict['test_shift'], old_labels, old_labels_remove_rate)
        else:
            val, test = ds_dict['val_shift'], ds_dict['test_shift']

        # sub_mkt, _ = self.split(ds_dict['market'], new_labels + old_labels, 0.08) # for acquisition estimator

        return {
            # 'train': old_aug_train,
            'train': old_train,
            'val': old_val,
            'test': old_test,
            # 'val_reg': val,
            # 'val_shift': sub_mkt,
            'val_shift': val,
            'test_shift': test,
            'market': ds_dict['market'],
            # 'aug_market': ds_dict['aug_market'],
        }, {
            'train': self.get_raw_indices(old_train),
            'val': self.get_raw_indices(old_val),
            'test': self.get_raw_indices(old_test),
            # 'val_reg': val,
            # 'val_shift': sub_mkt,
            'val_shift': self.get_raw_indices(val),
            'test_shift': self.get_raw_indices(test),
            'market': self.get_raw_indices(ds_dict['market']),           
        }
   
    def cover_load(self, ds_dict, remove_rate, cover_labels):
        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']
        logger.info('Cover Labels remove, src:{}, target: {}'.format(cover_labels['src'], cover_labels['target']))

        old_train, _ = self.split(ds_dict['train'], cover_labels['src'], new_labels_remove_rate)
        old_val, _ = self.split(ds_dict['val_shift'], cover_labels['src'], new_labels_remove_rate)
        old_test, _ = self.split(ds_dict['test_shift'], cover_labels['src'], new_labels_remove_rate)
        val_shift, _ = self.split(ds_dict['val_shift'], cover_labels['target'], new_labels_remove_rate)
        test_shift, _ = self.split(ds_dict['test_shift'], cover_labels['target'], new_labels_remove_rate)

        if old_labels_remove_rate != None:
            old_labels = cover_labels['src']
            _, val = self.split(val_shift, old_labels, old_labels_remove_rate)
            _, test = self.split(test_shift, old_labels, old_labels_remove_rate)
        else:
            val, test = val_shift, test_shift
        
        return {
            'train': old_train,
            'val': old_val,
            'test': old_test,
            'val_shift': val,
            'test_shift': test,
            'market': ds_dict['market'],
            # 'aug_market': ds_dict['aug_market'],
        }, {
            'train': self.get_raw_indices(old_train),
            'val': self.get_raw_indices(old_val),
            'test': self.get_raw_indices(old_test),
            # 'val_reg': val,
            # 'val_shift': sub_mkt,
            'val_shift': self.get_raw_indices(val),
            'test_shift': self.get_raw_indices(test),
            'market': self.get_raw_indices(ds_dict['market']),           
        }

    def create(self, config, raw_ds:Dict[str, cifar.data.Dataset]):
        data_config = config['data']
        label_config = data_config['labels']
        select_fine_labels = label_config['select_fine_labels']
        max_subclass_num = config['hparams']['subclass']

        ratio = data_config['ratio']
        train_size = ratio["train"]
        test_size = ratio["test"]
        val_from_test = ratio['val_from_test']

        label_summary = [i for i in range(max_subclass_num)] if len(select_fine_labels)==0 else select_fine_labels

        train_market, test_val = raw_ds['train_market'], raw_ds['test_val']
        train_ds, market_ds = self.split(train_market, label_summary, train_size)

        # aug_train_ds = torch.utils.data.Subset(aug_train_market, split_indices['target'])

        # aug_market_ds = torch.utils.data.Subset(aug_train_market, split_indices['other'])

        test_val, _ = self.split(test_val, label_summary, test_size)

        val_ds, test_ds = self.split(test_val, label_summary, val_from_test)

        raw_indices = {
            'val_shift': self.get_raw_indices(val_ds),
            'market': self.get_raw_indices(market_ds),
            'test_shift': self.get_raw_indices(test_ds),
            'train': self.get_raw_indices(train_ds)
        }
        return raw_indices

    def split(self, dataset, labels, target_ratio, use_fine_label = True, return_split_indices=False):
        '''
        Split Dataset by Selecting Data of Labels with target_ratio\n
        Return (target dataset, other dataset)
        '''
        if target_ratio is None:
            return dataset, None
        
        dataset_labels = self.get_labels(dataset, use_fine_label)

        target_splits, other_splits = self.get_split_indices(dataset_labels, labels, target_ratio)
        target_subset = torch.utils.data.Subset(dataset, target_splits)
        other_subset = torch.utils.data.Subset(dataset, other_splits)

        if return_split_indices:
            split_indices = {'target': target_splits, 'other': other_splits}
            return target_subset, other_subset, split_indices
        else:
            return target_subset, other_subset

    def fine_label_raw_subset(self, raw_ds:Dict[str, cifar.CIFAR100], select_fine_labels):
        '''
        Raw subset from selected fine labels
        '''
        train_market, test_val = raw_ds['train_market'], raw_ds['test_val']

        train_market, _ = self.get_subset_by_labels(train_market, select_fine_labels)
        test_val, _ = self.get_subset_by_labels(test_val, select_fine_labels)
        # aug_train_market = torch.utils.data.Subset(aug_train_market, select_indices)  

        return {
            'train_market': train_market,
            'test_val': test_val,
            # 'aug_train_market': aug_train_market
        }
        
    def get_raw_dataset(self, dataset_root, normalize_stat, label_map):
        '''
        Get raw dataset
        '''

        mean, std = normalize_stat['mean'], normalize_stat['std']

        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        # augment_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     base_transform
        # ])

        train_market = cifar.CIFAR100(dataset_root, train=True,img_transform=base_transform, coarse_label_transform=label_map)
        # aug_train_market = cifar.CIFAR100(dataset_root, train=True,img_transform=augment_transform, coarse_label_transform=label_map)
        test_val = cifar.CIFAR100(dataset_root, train=False,img_transform=base_transform, coarse_label_transform=label_map)

        return {
            'train_market': train_market,
            'test_val': test_val,
            # 'aug_train_market': aug_train_market
        }
    
    def get_labels(self, ds, use_fine_label):
        ds_size = len(ds)
        label_idx = 2 if use_fine_label else 1
        labels = []
        for idx in range(ds_size):
            label = ds[idx][label_idx]
            labels.append(label)
        return np.array(labels)

    def create_4_splits(self, epochs, config):
        data_config = config['data']
        label_config = data_config['labels']
        select_fine_labels = label_config['select_fine_labels']
        label_map = label_config['map']
        dataset_root = data_config['root']
        
        mean, std = self.normalize(select_fine_labels, dataset_root)
        normalize_stat = {
            'mean': mean,
            'std': std
        }

        raw_dataset = self.get_raw_dataset(dataset_root, normalize_stat, label_map)

        if len(select_fine_labels)>0:
            raw_dataset = self.fine_label_raw_subset(raw_dataset, select_fine_labels)

        raw_indices_list = []
        for epo in range(epochs):
            dataset_raw_indices = self.create(config, raw_dataset)
            raw_indices_list.append(dataset_raw_indices)

        return raw_indices_list, normalize_stat

    def normalize(self, select_fine_labels, dataset_root):
        train_ds = cifar.CIFAR100(dataset_root, train=True)
        if len(select_fine_labels) > 0:
            train_ds, _ = self.get_subset_by_labels(train_ds, select_fine_labels)
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
            color = torch.concat(color) # color list: n_sample*[(H, W)]
            mean.append(torch.mean(color).item())
            std.append(torch.std(color).item())
        return mean, std #superclass

class Core(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def split(self, dataset, target_ratio, option: dict, return_splits=False):
        '''
        Split Dataset by Target Labels \n
        Return (target dataset, the rest)
        '''

        option_name = option['name']
        option_labels = option['labels']
        dataset_labels = self.get_labels(dataset, option_name)

        target_splits, other_splits = self.get_split_indices(dataset_labels, option_labels, target_ratio)

        target_subset = torch.utils.data.Subset(dataset, target_splits)
        other_subset = torch.utils.data.Subset(dataset, other_splits)

        if return_splits:
            return target_subset, other_subset, {'taregt': target_splits, 'other': other_splits}
        else:
            return target_subset, other_subset
        
    def normalize(self, meta_data:MetaData):
        train_ds = meta_data.data
        r, g, b = [], [], []
        train_size = len(train_ds)
        for idx in range(train_size):
            raw_img = train_ds[idx]
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

    def primal_operate(self, ds_dict, remove_rate, remove_label_config, option):

        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']

        subclass = self.subclass_config(remove_label_config, option)
        logger.info('option: {}, Removed Labels:{}'.format(option, subclass['new']))

        # _, old_aug_train = self.split(ds_dict['aug_train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_train = self.split(ds_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        # old_aug_train = torch.utils.data.Subset(ds_dict['aug_train'], split_indices['other'])

        _, old_val = self.split(ds_dict['val_shift'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_test = self.split(ds_dict['test_shift'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        
        if old_labels_remove_rate != None:
            _, val = self.split(ds_dict['val_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
            _, test = self.split(ds_dict['test_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
        else:
            val, test = ds_dict['val_shift'], ds_dict['test_shift']

        # sub_mkt, _ = self.split(ds_dict['market'], subclass['new'] + subclass['old'], 0.1)

        return {
            # 'train': old_aug_train,
            'train': old_train,
            'val': old_val,
            'test': old_test,
            'val_shift': val,
            'test_shift': test,
            'market': ds_dict['market'],
            # 'aug_market': ds_dict['aug_market'],
        }, {
            'train': self.get_raw_indices(old_train),
            'val': self.get_raw_indices(old_val),
            'test': self.get_raw_indices(old_test),
            'val_shift': self.get_raw_indices(val),
            'test_shift': self.get_raw_indices(test),
            'market': self.get_raw_indices(ds_dict['market']),            
        }

    def dual_operate(self, primal_dict, remove_rate, remove_label_config, option):

        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']
        
        subclass = self.subclass_config(remove_label_config, option)
        logger.info('option: {}, Removed Labels: {}'.format(option, subclass['new']))

        # _, old_aug_train = self.split(primal_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_train = self.split(primal_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        # old_aug_train = torch.utils.data.Subset(primal_dict['train'], split_indices['other'])
        
        _, old_val = self.split(primal_dict['val'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_test = self.split(primal_dict['test'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        
        if old_labels_remove_rate != None:
            _, val = self.split(primal_dict['val_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
            _, test = self.split(primal_dict['test_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
        else:
            val, test = primal_dict['val_shift'], primal_dict['test_shift']
            
        return {
            'train': old_train,
            'val': old_val,
            'test': old_test,
            'val_shift': val,
            'test_shift': test,
            'market': primal_dict['market'],
        }, {
            'train': self.get_raw_indices(old_train),
            'val': self.get_raw_indices(old_val),
            'test': self.get_raw_indices(old_test),
            'val_shift': self.get_raw_indices(val),
            'test_shift': self.get_raw_indices(test),
            'market': self.get_raw_indices(primal_dict['market']),            
        }
        

    def create_shift(self, ds_dict, remove_rate, label_config, option):
        remove_label_config = label_config['remove']
        if option == 'both':
            primal_dict = self.primal_operate(ds_dict, remove_rate, remove_label_config, option = 'session')
            return self.dual_operate(primal_dict, remove_rate, remove_label_config, option='object')
        else:
            return self.primal_operate(ds_dict, remove_rate, remove_label_config, option)

    def choose_label_index(self, option):
        if option == 'object':
            return 2
        elif option == 'session':
            return 3
        else:
            return 1
        
    def get_labels(self, ds, option:str):
        ds_size = len(ds)
        label_idx = self.choose_label_index(option)
        labels = []
        for idx in range(ds_size):
            label = ds[idx][label_idx]
            labels.append(label)
        return np.array(labels)
        
    def create(self, data_config, meta: MetaData):

        ratio = data_config['ratio']
        train_size = ratio["train"]
        test_size = ratio["test_from_all"]
        val_from_test = ratio['val_from_test']

        label_map = data_config['labels']['map']
        categories = list(label_map.keys()) if label_map != None else [i for i in range(10)]
        sessions = [i for i in range(11)]

        all_data_indices = np.arange(len(meta.data))

        test_train_split = meta.balanced_split(test_size, categories, sessions, all_data_indices)
        
        test_val_indices = test_train_split['sampled']
        train_market_indices = test_train_split['others']

        test_val_raw_indices = all_data_indices[test_val_indices]
        train_market_raw_indices = all_data_indices[train_market_indices]

        val_test_split = meta.balanced_split(val_from_test, categories, sessions, test_val_raw_indices)
        # val_dict = self.subset2dict(val_test_split['sampled'], test_val_indices)
        # test_dict = self.subset2dict(val_test_split['others'], test_val_indices)

        train_market_split = meta.balanced_split(train_size, categories, sessions, train_market_raw_indices)
        # train_dict = self.subset2dict(train_market_split['sampled'], train_market_indices)
        # market_dict = self.subset2dict(train_market_split['others'], train_market_indices)

        raw_indices ={
            'val_shift': test_val_raw_indices[val_test_split['sampled']],
            'test_shift': test_val_raw_indices[val_test_split['others']],
            'train': train_market_raw_indices[train_market_split['sampled']],
            'market': train_market_raw_indices[train_market_split['others']],
        }
        return raw_indices

    def subclass_config(self, remove_config, option = 'session'):
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
        
    def get_raw_dataset(self, meta: MetaData, normalize_stat, label_map):

        meta_dict = {
            'data': meta.data,
            'session': meta.session_labels,
            'object': meta.object_labels,
            'category': meta.category_labels
        }
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_stat['mean'], std=normalize_stat['std']),
            ])
        
        # augment_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     base_transform
        # ])
        meta_dataset = core.Core(meta_dict, base_transform, label_map)
        # aug_meta_dataset = core.Core(meta_dict, augment_transform, label_map)
        return meta_dataset
    
    def load_dataset_raw_indices(self, raw_indices_dict: Dict[str, list], data_config, normalized_stat):
        sampled_meta = MetaData(data_config['root'])
        raw_dataset = self.get_raw_dataset(sampled_meta, normalized_stat, data_config['labels']['map'])
        indices_dict = {
            'val_shift': torch.utils.data.Subset(raw_dataset, raw_indices_dict['val_shift']),
            'test_shift': torch.utils.data.Subset(raw_dataset, raw_indices_dict['test_shift']),
            'train': torch.utils.data.Subset(raw_dataset, raw_indices_dict['train']),
            'market': torch.utils.data.Subset(raw_dataset, raw_indices_dict['market']), 
            # 'aug_train': torch.utils.data.Subset(raw_dataset['aug_meta'], raw_indices_dict['train']), 
            # 'aug_market': torch.utils.data.Subset(raw_dataset['aug_meta'], raw_indices_dict['market']), 
        }
        if len(list(raw_indices_dict.keys()))>4:
            indices_dict['val'] = torch.utils.data.Subset(raw_dataset, raw_indices_dict['val'])
            indices_dict['test'] = torch.utils.data.Subset(raw_dataset, raw_indices_dict['test'])
        return indices_dict       

    def create_4_splits(self, epochs, config):
        data_config = config['data']

        sampled_meta = MetaData(config['data']['root'])

        mean, std = self.normalize(sampled_meta)

        normalize_stat = {
            'mean': mean,
            'std': std
        }

        raw_indices_list = []
        for epo in range(epochs):
            raw_indices = self.create(data_config, sampled_meta)
            raw_indices_list.append(raw_indices)
        return raw_indices_list, normalize_stat
    
    def get_indices_by_labels(self, ds, subset_labels, option):
        select_indice = []
        ds_labels = self.get_labels(ds, option)
        for c in subset_labels:
            select_indice.append(np.arange(len(ds))[c==ds_labels])
        select_indice = np.concatenate(select_indice)
        return select_indice

    def get_subset_by_labels(self, ds, subset_labels, option, return_split=False):
        select_indice = self.get_indices_by_labels(ds, subset_labels, option)
        ds = torch.utils.data.Subset(ds,select_indice) 
        if return_split:
            return ds, select_indice
        else:
            return ds, None

def factory(name):
    if name == 'core':
        return Core()
    else:
        return Cifar()

