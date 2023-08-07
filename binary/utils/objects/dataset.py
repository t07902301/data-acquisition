import torchvision.transforms as transforms
from utils import config
import torch
import numpy as np
from utils.env import generator
import utils.objects.cifar as cifar
import utils.objects.Config as Config

data_config = config['data']

max_subclass_num = config['hparams']['subclass']

class DataSplits():
    dataset: dict
    loader: dict
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        generator.manual_seed(0)    
        self.loader = {
            k: torch.utils.data.DataLoader(self.dataset[k], batch_size=batch_size, 
                                        shuffle=(k=='train'), drop_last=(k=='train'),
                                        num_workers=config['num_workers'],generator=generator)
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
                                                              num_workers=config['num_workers'],generator=generator)
        
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

def load_dataset(ds_dict, remove_rate, remove_labels, old_labels=None):
    train_remove_rate = remove_rate['train']
    test_remove_rate = remove_rate['test']
    market_remove_rate = remove_rate['market']
    _, old_train = split_dataset(ds_dict['train'], remove_labels, train_remove_rate)
    _, old_val = split_dataset(ds_dict['val_shift'], remove_labels, train_remove_rate)
    _, old_test = split_dataset(ds_dict['test_shift'], remove_labels, train_remove_rate)
    
    if test_remove_rate != None:
        _, val = split_dataset(ds_dict['val_shift'], old_labels, test_remove_rate)
        _, test = split_dataset(ds_dict['test_shift'], old_labels, test_remove_rate)
    else:
        val, test = ds_dict['val_shift'], ds_dict['test_shift']
    
    if market_remove_rate != None:
        _, market = split_dataset(ds_dict['market'], remove_labels, market_remove_rate)
    else:
        market = ds_dict['market']

    return {
        'train': old_train,
        'train_non_cnn': old_train,
        'val': old_val,
        'test': old_test,
        'val_shift': val,
        'test_shift': test,
        'market': market
    }

def load_cover_dataset(ds_dict, remove_rate, cover_labels, old_labels=None):
    train_remove_rate = remove_rate['train']
    test_remove_rate = remove_rate['test']
    market_remove_rate = remove_rate['market']

    old_train, _ = split_dataset(ds_dict['train'], cover_labels['src'], train_remove_rate)
    old_val, _ = split_dataset(ds_dict['val_shift'], cover_labels['src'], train_remove_rate)
    old_test, _ = split_dataset(ds_dict['test_shift'], cover_labels['src'], train_remove_rate)
    val_shift, _ = split_dataset(ds_dict['val_shift'], cover_labels['target'], train_remove_rate)
    test_shift, _ = split_dataset(ds_dict['test_shift'], cover_labels['target'], train_remove_rate)

    market_target, _ = split_dataset(ds_dict['market'], cover_labels['target'], 1.0)

    if test_remove_rate != None:
        _, val = split_dataset(val_shift, old_labels, test_remove_rate)
        _, test = split_dataset(test_shift, old_labels, test_remove_rate)
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
        # 'market': market_target,
        # 'market_target': market_target
    }

def create_dataset(select_fine_labels, ratio, mean, std):
    # When all classes are used, only work on removal
    # When some classes are neglected, test set and the big train set will be shrank.
    train_ds, test_ds = get_raw_ds(data_config['ds_root'], mean, std)
    train_size = ratio["train_size"]
    market_size = ratio["market_size"]

    if len(select_fine_labels)>0:
        train_ds = get_subset_by_labels(train_ds, select_fine_labels)
        test_ds = get_subset_by_labels(test_ds, select_fine_labels)

    label_summary = [i for i in range(max_subclass_num)] if len(select_fine_labels)==0 else select_fine_labels
    
    train_ds, market_ds = split_dataset(train_ds, label_summary, train_size/ (train_size + market_size) )

    test_ds, val_ds = split_dataset(test_ds, label_summary, 0.5)

    ds = {}
    # modified_labels = list(set(select_fine_labels) - set(target_test_label))
    # balanced_train_ds = balance_dataset(target_test_label, modified_labels, left_train) # make shifted and original labels balanced?
    ds['val_shift'] =  val_ds
    ds['market'] =  market_ds
    ds['test_shift'] = test_ds
    ds['train'] =  train_ds
    return ds

def split_dataset(dataset, target_labels, split_1_ratio, use_fine_label = True):
    '''
    Split Dataset by Selecting Data from Target Labels \n
    Return (target dataset, the rest)
    '''
    dataset_labels = get_labels(dataset, use_fine_label)
    splits_1, splits_2 = get_split_indices(dataset_labels, target_labels, split_1_ratio)
    subset_1 = torch.utils.data.Subset(dataset, splits_1)
    subset_2 = torch.utils.data.Subset(dataset, splits_2)
    return subset_1, subset_2

def balance_dataset(target_labels, modified_label, dataset):
    target_ds = get_subset_by_labels(dataset, target_labels)
    n_modify_cls = int(len(target_ds)/len(modified_label))
    ds_labels = get_labels(dataset)
    ds_indices = np.arange(len(dataset))
    modified_indices = []
    for label in modified_label:
        cls_mask = (ds_labels==label)
        sampled = sample_indices(ds_indices[cls_mask], n_modify_cls)
        modified_indices.append(sampled)
    modified_indices = np.concatenate(modified_indices)
    modified_ds = torch.utils.data.Subset(dataset, modified_indices)
    return torch.utils.data.ConcatDataset([modified_ds, target_ds])

def modify_coarse_label(dataset, label_map):
    for split in dataset.keys():
        dataset[split] = cifar.ModifiedDataset(dataset=dataset[split],coarse_label_transform=label_map)
    return dataset

def get_vis_transform(mean, std):
    # For visualization
    INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/x for x in std]),
                                transforms.Normalize(mean = [-x for x in mean],
                                                     std = [ 1., 1., 1. ]),])
    TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])
    return INV_NORM, TOIMAGE

def get_raw_ds(ds_root, mean, std):
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
    # augment_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     base_transform
    # ])
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    # aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=augment_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    return train_ds,test_ds

def get_indices_by_labels(ds, subset_labels, use_fine_label=True):
    select_indice = []
    ds_labels = get_labels(ds,use_fine_label)
    for c in subset_labels:
        select_indice.append(np.arange(len(ds))[c==ds_labels])
    select_indice = np.concatenate(select_indice)
    return select_indice

def get_subset_by_labels(ds, subset_labels, use_fine_label=True):
    select_indice = get_indices_by_labels(ds, subset_labels, use_fine_label)
    ds = torch.utils.data.Subset(ds,select_indice) 
    return ds

def get_labels(ds,use_fine_label=True):
    ds_size = len(ds)
    label_idx = 2 if use_fine_label else 1
    labels = []
    for idx in range(ds_size):
        label = ds[idx][label_idx]
        labels.append(label)
    return np.array(labels)
    
def sample_indices(indices,ratio):
    if type(ratio) == int:
        return np.random.choice(indices,ratio,replace=False)
    else:
        return np.random.choice(indices,int(ratio*len(indices)),replace=False)

def complimentary_mask(mask_length, active_spot):
    '''
    reverse the active mask
    '''
    active_mask = np.zeros(mask_length, dtype=bool)
    active_mask[active_spot] = True
    advert_mask = ~active_mask 
    return advert_mask

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

def get_data_splits_list(epochs, select_fine_labels, label_map, ratio):
    mean, std = normalize()
    normalize_stat = {
        'mean': mean,
        'std': std
    }
    ds_list = []
    for epo in range(epochs):
        ds = create_dataset(select_fine_labels,ratio, mean, std)
        if len(select_fine_labels) != 0 and (isinstance(label_map, dict)):
            ds = modify_coarse_label(ds, label_map)
        ds_list.append(ds)
    return ds_list, normalize_stat

def check_labels(epochs, select_fine_labels, label_map, ratio):
    for epo in range(epochs):
        ds = create_dataset(select_fine_labels,ratio)
        for key, value in ds.items():
            labels = get_labels(value, False)
            labels_sum = np.unique(labels)
            print(key, 'coarse labels :', labels_sum)

def normalize():
    train_ds = cifar.CIFAR100(data_config['ds_root'], train=True, coarse=True)
    select_fine_labels = data_config['select_fine_labels']
    if len(select_fine_labels) > 0:
        train_ds = get_subset_by_labels(train_ds, select_fine_labels)
    r, g, b = [], [], []
    train_size = len(train_ds)
    for idx in range(train_size):
        img = transforms.ToTensor()(train_ds[idx][0])
        r.append(img[0])
        g.append(img[1])
        b.append(img[2])
    result = [ r, g, b]
    mean = []
    std = []
    for color in result:
        color = torch.stack(color)
        mean.append(torch.mean(color).item())
        std.append(torch.std(color).item())
    return mean, std