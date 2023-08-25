import torchvision.transforms as transforms
import torch
from utils.env import generator
import utils.objects.Config as Config
from abc import abstractmethod
import utils.dataset.cifar as cifar
import utils.dataset.core as core
import pickle as pkl
import numpy as np
import time

def complimentary_mask(mask_length, active_spot):
    '''
    reverse the active mask
    '''
    active_mask = np.zeros(mask_length, dtype=bool)
    active_mask[active_spot] = True
    advert_mask = ~active_mask 
    return advert_mask

def get_vis_transform(mean, std):
    # For visualization
    INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/x for x in std]),
                                transforms.Normalize(mean = [-x for x in mean],
                                                    std = [ 1., 1., 1. ]),])
    TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])
    return INV_NORM, TOIMAGE

n_workers = 1

class DataSplits():
    dataset: dict
    loader: dict
    def __init__(self, dataset, batch_size, name=None) -> None:
        self.dataset = dataset
        self.dataset_name = name
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

class Dataset():
    def __init__(self) -> None:
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
    def load(self, ds_dict, remove_rate, label_config, option):
        '''
        Load dataset splits with data shift degree
        '''
        pass

    @abstractmethod
    def create(self):
        '''
        Create dataset splits by ratios from config
        '''
        pass

    @abstractmethod
    def get_data_splits_list(self, epochs, config, meta_path):
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

        if len(target_labels) == 0 or split_1_ratio == 0:
            return None, np.arange(ds_length)
        
        p1_indices = []

        for c in target_labels:
            cls_indices = np.arange(ds_length)[dataset_labels == c]
            split_indices = self.sample_indices(cls_indices,split_1_ratio)
            p1_indices.append(split_indices)

        p1_indices = np.concatenate(p1_indices)
        mask_p2 = complimentary_mask(mask_length=ds_length,active_spot=p1_indices)
        p2_indices = np.arange(ds_length)[mask_p2]
        return p1_indices,p2_indices

class Cifar(Dataset):
    
    def __init__(self) -> None:
        super().__init__()

    def get_indices_by_labels(self, ds, subset_labels):
        select_indice = []
        ds_labels = self.get_labels(ds, use_fine_label=True)
        for c in subset_labels:
            select_indice.append(np.arange(len(ds))[c==ds_labels])
        select_indice = np.concatenate(select_indice)
        return select_indice

    def get_subset_by_labels(self, ds, subset_labels, return_indices=False):
        select_indice = self.get_indices_by_labels(ds, subset_labels)
        ds = torch.utils.data.Subset(ds,select_indice) 
        if return_indices:
            return ds, select_indice
        else:
            return ds, None
        
    def load(self, ds_dict, remove_rate, label_config, option):
        if 'cover' in label_config:
            return self.cover_load(ds_dict, remove_rate, label_config['cover'])
        else:
            return self.non_cover_load(ds_dict, remove_rate, label_config, option)

    def non_cover_load(self, ds_dict, remove_rate, label_config, option):
        new_labels = label_config['remove']
        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']
        old_labels = list(set(label_config['select_fine_labels']) - set(new_labels))
        print('Fine Labels removed', new_labels)

        _, old_aug_train = self.split(ds_dict['aug_train'], new_labels, new_labels_remove_rate)
        _, old_train = self.split(ds_dict['train'], new_labels, new_labels_remove_rate)
        _, old_val = self.split(ds_dict['val_shift'], new_labels, new_labels_remove_rate)
        _, old_test = self.split(ds_dict['test_shift'], new_labels, new_labels_remove_rate)
        
        if old_labels_remove_rate != None:
            _, val = self.split(ds_dict['val_shift'], old_labels, old_labels_remove_rate)
            _, test = self.split(ds_dict['test_shift'], old_labels, old_labels_remove_rate)
        else:
            val, test = ds_dict['val_shift'], ds_dict['test_shift']

        # sub_mkt, _ = self.split(ds_dict['market'], new_labels + old_labels, 0.1)

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
   
    def cover_load(self, ds_dict, remove_rate, cover_labels):
        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']

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
            'train_non_cnn': old_train,
            'market': ds_dict['market'],
        }

    def create(self, config, raw_ds:dict):
        # When all classes are used, only work on removal
        # When some classes are neglected, test set and the big train set will be shrank.
        data_config = config['data']
        label_config = data_config['labels']
        select_fine_labels = label_config['select_fine_labels']
        max_subclass_num = config['hparams']['subclass']

        ratio = data_config['ratio']
        train_size = ratio["train"]
        test_size = ratio["test"]
        val_from_test = ratio['val_from_test']

        label_summary = [i for i in range(max_subclass_num)] if len(select_fine_labels)==0 else select_fine_labels

        train_market, test_val, aug_train_market = raw_ds['train_market'], raw_ds['test_val'], raw_ds['aug_train_market']
        
        train_ds, market_ds, split_indices = self.split(train_market, label_summary, train_size, return_split_indices=True)

        aug_train_ds = torch.utils.data.Subset(aug_train_market, split_indices['target'])

        aug_market_ds = torch.utils.data.Subset(aug_train_market, split_indices['other'])

        test_val, _ = self.split(test_val, label_summary, test_size)

        val_ds, test_ds = self.split(test_val, label_summary, val_from_test)

        ds = {}
        # modified_labels = list(set(select_fine_labels) - set(target_test_label))
        # balanced_train_ds = balance_dataset(target_test_label, modified_labels, left_train) # make shifted and original labels balanced?
        ds['val_shift'] =  val_ds
        ds['market'] =  market_ds
        ds['test_shift'] = test_ds
        ds['train'] =  train_ds
        ds['aug_market'] = aug_market_ds
        ds['aug_train'] = aug_train_ds
        return ds

    def split(self, dataset, labels, target_ratio, use_fine_label = True, return_split_indices=False):
        '''
        Split Dataset by Selecting Data from Labels with target_ratio\n
        Return (target dataset, other dataset)
        '''
        if target_ratio == None:
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

    def modify_coarse_label(self,dataset, label_map):
        for split in dataset.keys():
            dataset[split] = cifar.ModifiedDataset(dataset=dataset[split],coarse_label_transform=label_map)
        return dataset

    def get_raw_ds(self, ds_root, mean, std, select_fine_labels):
        '''
        Get raw dataset with subset selected for target classes
        '''
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            base_transform
        ])

        train_market = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
        aug_train_market = cifar.CIFAR100(ds_root, train=True,transform=augment_transform,coarse=True)
        test_val = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)

        if len(select_fine_labels)>0:
            train_market, select_indices = self.get_subset_by_labels(train_market, select_fine_labels, return_indices=True)
            test_val, _ = self.get_subset_by_labels(test_val, select_fine_labels)
            aug_train_market = torch.utils.data.Subset(aug_train_market, select_indices)  

        return {
            'train_market': train_market,
            'test_val': test_val,
            'aug_train_market': aug_train_market
        }
    
    def get_labels(self, ds,use_fine_label):
        ds_size = len(ds)
        label_idx = 2 if use_fine_label else 1
        labels = []
        for idx in range(ds_size):
            label = ds[idx][label_idx]
            labels.append(label)
        return np.array(labels)

    def get_data_splits_list(self, epochs, config, meta_path):
        data_config = config['data']
        label_config = data_config['labels']
        select_fine_labels = label_config['select_fine_labels']
        label_map = label_config['map']
        dataset_root = data_config['ds_root']
        
        mean, std = self.normalize(select_fine_labels, dataset_root)
        normalize_stat = {
            'mean': mean,
            'std': std
        }

        raw_ds = self.get_raw_ds(data_config['ds_root'], mean, std, select_fine_labels)

        ds_list = []
        for epo in range(epochs):
            ds = self.create(config, raw_ds)
            if len(select_fine_labels) != 0 and (isinstance(label_map, dict)):
                ds = self.modify_coarse_label(ds, label_map)
            ds_list.append(ds)

        return ds_list, normalize_stat

    def normalize(self, select_fine_labels, ds_root):
        train_ds = cifar.CIFAR100(ds_root, train=True, coarse=True)
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

    def split(self, dataset, target_ratio, option: dict):
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

        return target_subset, other_subset

    def modify_coarse_label(self):
        pass

    def normalize(self, meta_data):
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

    def primal_operate(self, ds_dict, remove_rate, remove_label_config, option):

        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']

        subclass = self.subclass_config(remove_label_config, option)
        print(option, 'Removed Labels:', subclass['new'])

        _, old_aug_train = self.split(ds_dict['aug_train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_train = self.split(ds_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_val = self.split(ds_dict['val'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_test = self.split(ds_dict['test'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        
        if old_labels_remove_rate != None:
            _, val = self.split(ds_dict['val'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
            _, test = self.split(ds_dict['test'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
        else:
            val, test = ds_dict['val'], ds_dict['test']

        # sub_mkt, _ = self.split(ds_dict['market'], subclass['new'] + subclass['old'], 0.1)

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

    def dual_operate(self, primal_dict, remove_rate, remove_label_config, option):

        new_labels_remove_rate = remove_rate['new_labels']
        old_labels_remove_rate = remove_rate['old_labels']
        
        subclass = self.subclass_config(remove_label_config, option)
        print(option, 'Removed Labels:', subclass['new'])

        _, old_aug_train = self.split(primal_dict['train'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_train = self.split(primal_dict['train_non_cnn'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_val = self.split(primal_dict['val'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        _, old_test = self.split(primal_dict['test'], new_labels_remove_rate, {'name': option, 'labels': subclass['new']})
        
        if old_labels_remove_rate != None:
            _, val = self.split(primal_dict['val_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
            _, test = self.split(primal_dict['test_shift'], old_labels_remove_rate, {'name': option, 'labels': subclass['old']})
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

    def load(self, ds_dict, remove_rate, label_config, option):
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
        
    def create(self, meta_split:dict, mean, std, label_map):
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
        
    def get_data_splits_list(self, epochs, config, meta_path):

        data_config = config['data']
        label_map = data_config['labels']['map']

        meta_data = MetaData(meta_path)

        mean, std = self.normalize(meta_data)

        normalize_stat = {
            'mean': mean,
            'std': std
        }
        ds_list = []
        for epo in range(epochs):
            meta_split = meta_data.split(data_config)
            ds = self.create(meta_split, mean, std, label_map)
            ds_list.append(ds)
        return ds_list, normalize_stat
    
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

    def balanced_split(self, ratio, categories, sessions):
        '''
        Split a Dataset Dict balanced from category, session to object. Ratio is to sample in object levels.
        '''

        dataset_size = len(self.data)
        
        indices = np.arange(dataset_size)

        sampled_cat_indices = []

        for cat in categories:

            cat_indices = indices[self.category_labels==cat]
            cat_session_labels = self.session_labels[cat_indices]

            objects = [i + cat * 5 for i in range(5)]

            sampled_session_indices = []
            for session in sessions: # session labels given a cat

                cat_session_indices = cat_indices[cat_session_labels==session]
                cat_session_obj_labels = self.object_labels[cat_session_indices]

                subset_indices = self.get_subset(cat_session_obj_labels, objects, cat_session_indices, ratio) # obj labels given a session
                sampled_session_indices.append(subset_indices)
            
            sampled_session_indices = np.concatenate(sampled_session_indices, axis = 0)

            sampled_cat_indices.append(sampled_session_indices)

        sampled_cat_indices = np.concatenate(sampled_cat_indices, axis = 0)

        return {
            'sampled': sampled_cat_indices,
            'others': complimentary_mask(dataset_size, sampled_cat_indices)
        }

    def subset2dict(self, subset_indices):
        return {
            'data': self.data[subset_indices],
            'session': self.session_labels[subset_indices],
            'object': self.object_labels[subset_indices],
            'category': self.category_labels[subset_indices]
        }

    def split(self, data_config):
        '''
        Get train, market, val, test set without any shift
        '''
        ratio = data_config['ratio']
        train_size = ratio["train"]
        test_size = ratio["test"]
        val_from_test = ratio['val_from_test']

        label_map = data_config['labels']['map']
        categories = list(label_map.keys()) if label_map != None else [i for i in range(10)]
        sessions = [i for i in range(11)]

        test_train_split = self.balanced_split(test_size, categories, sessions)
        test_val_dict = self.subset2dict(test_train_split['sampled'])
        train_market_dict = self.subset2dict(test_train_split['others'])

        train_market_split = self.balanced_split(train_size, train_market_dict, categories, sessions)
        train_dict = self.subset2dict(train_market_dict, train_market_split['sampled'])
        market_dict = self.subset2dict(train_market_dict, train_market_split['others'])

        val_test_split = self.balanced_split(val_from_test, test_val_dict, categories, sessions)
        val_dict = self.subset2dict(test_val_dict, val_test_split['sampled'])
        test_dict = self.subset2dict(test_val_dict, val_test_split['others'])

        ds = {}
        ds['val'] =  val_dict
        ds['market'] =  market_dict
        ds['test'] = test_dict
        ds['train'] =  train_dict
        return ds
