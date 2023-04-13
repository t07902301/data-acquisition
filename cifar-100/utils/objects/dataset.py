import torchvision.transforms as transforms
from utils import config
import torch
import numpy as np
from utils.env import generator, data_split_env
import utils.objects.cifar as cifar
import utils.objects.Config as Config
data_config = config['data']
max_subclass_num = config['hparams']['subclass']
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=np.array(data_config['mean'])/255, std=np.array(data_config['std'])/255)])
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    base_transform
])
class DataSplits():
    dataset: dict
    loader: dict
    def __init__(self,ds_root,select_fine_labels,ratio) -> None:
        self.dataset = self.create_dataset_split(ds_root,select_fine_labels,ratio)
    def create_dataset_split(self, ds_root, select_fine_labels, ratio):
        # When all classes are used, only work on removal
        # When some classes are neglected, test set and the big train set will be shrank.
        train_ds,aug_train_ds,test_ds = get_raw_ds(ds_root)

        train_size = ratio["train_size"]
        val_size = ratio["val_size"]
        market_size = ratio["market_size"]
        remove_rate = ratio['remove_rate']

        if len(select_fine_labels)>0:
            train_ds = get_subset_by_labels(train_ds,select_fine_labels)
            test_ds = get_subset_by_labels(test_ds,select_fine_labels)
            aug_train_ds = get_subset_by_labels(aug_train_ds,select_fine_labels)

        label_summary = [i for i in range(max_subclass_num)] if len(select_fine_labels)==0 else select_fine_labels

        train_fine_labels = get_ds_labels(train_ds)
        train_indices,val_market_indices = split_dataset(train_fine_labels, label_summary, train_size/(train_size+val_size+market_size))

        val_market_set = torch.utils.data.Subset(train_ds,val_market_indices)
        aug_val_market_set = torch.utils.data.Subset(aug_train_ds,val_market_indices)
        clip_train_ds_split = torch.utils.data.Subset(train_ds,train_indices)
        aug_train_ds_split = torch.utils.data.Subset(aug_train_ds,train_indices)

        val_mar_fine_labels = get_ds_labels(val_market_set)
        val_indices,market_indices = split_dataset(val_mar_fine_labels,label_summary,val_size/(val_size+market_size))

        val_ds = torch.utils.data.Subset(val_market_set,val_indices)
        market_ds = torch.utils.data.Subset(val_market_set,market_indices)
        aug_market_ds = torch.utils.data.Subset(aug_val_market_set,market_indices)

        split_train_fine_labels = get_ds_labels(aug_train_ds_split)
        _, left_indices = split_dataset(split_train_fine_labels,data_config['remove_fine_labels'],remove_rate)
        left_clip_train = torch.utils.data.Subset(clip_train_ds_split,left_indices)
        left_aug_train = torch.utils.data.Subset(aug_train_ds_split,left_indices)

        ds = {}
        ds['train'] =  left_aug_train
        ds['val'] =  val_ds
        ds['market'] =  market_ds
        ds['train_clip'] = left_clip_train
        ds['market_aug'] =  aug_market_ds
        ds['test'] = test_ds
        return ds
    def modify_coarse_label(self, label_map):
        for split in self.dataset.keys():
            self.dataset[split] = cifar.ModifiedDataset(dataset=self.dataset[split],coarse_label_transform=label_map)
    def expand(self, split_name, new_data, batch_size):
        '''
        dataset (dict) is mutable, pass by reference
        '''
        self.dataset[split_name] = torch.utils.data.ConcatDataset([self.dataset[split_name],new_data])
        self.update_dataloader(split_name, batch_size) 

    def reduce(self, split_name, remove_data_indices, batch_size):
        '''
        dataset (dict) is mutable, pass by reference
        '''
        original_split_len = len(self.dataset[split_name])
        left_market_mask = complimentary_mask(mask_length=original_split_len,active_spot=remove_data_indices)
        self.dataset[split_name] = torch.utils.data.Subset(self.dataset[split_name],np.arange(original_split_len)[left_market_mask])
        self.update_dataloader(split_name, batch_size) 

    def get_dataloader(self, batch_size):
        self.loader = {
            k: torch.utils.data.DataLoader(self.dataset[k], batch_size=batch_size, 
                                        shuffle=(k=='train'), drop_last=(k=='train'),num_workers=config['num_workers'],generator=generator)
            for k in self.dataset.keys()
        }
    def update_dataloader(self, split_name, batch_size):
        self.loader[split_name] = torch.utils.data.DataLoader(self.dataset[split_name], batch_size=batch_size, 
                                        shuffle=(split_name=='train'), drop_last=(split_name=='train'),num_workers=config['num_workers'],generator=generator)
    def update_dataset(self, split_name, updated_data, batch_size):
        self.dataset[split_name] = updated_data
        self.update_dataloader(split_name,batch_size)
    def use_new_data(self, new_data, new_model_config:Config.NewModel, acquisition_config:Config.Acquistion):
        '''
        new data to be added to train set or not, and update loader automatically
        '''
        assert len(new_data) == acquisition_config.get_new_data_size(new_model_config.class_number), 'size error - new data: {}, required new data: {} under {}'.format(len(new_data), acquisition_config.get_new_data_size(new_model_config.class_number), acquisition_config.get_info())

        if new_model_config.pure:
            self.update_dataset('train', new_data, new_model_config.batch_size)
        else:
            self.expand('train', new_data, new_model_config.batch_size)

def get_vis_transform(std,mean):
    # For visualization
    INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [255/x for x in std]),
                                    transforms.Normalize(mean = [-x /255 for x in mean],
                                                        std = [ 1., 1., 1. ])])
    TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])
    return INV_NORM, TOIMAGE

def get_raw_ds(ds_root):
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    return train_ds,aug_train_ds,test_ds

def get_subset_by_labels(ds, subset_labels, use_fine_label=True):
    select_indice = []
    ds_labels = get_ds_labels(ds,use_fine_label)
    for c in subset_labels:
        select_indice.append(np.arange(len(ds))[c==ds_labels])
    select_indice = np.concatenate(select_indice)
    ds = torch.utils.data.Subset(ds,select_indice) 
    return ds

def get_ds_labels(ds,use_fine_label=True):
    ds_size = len(ds)
    label_idx = 2 if use_fine_label else 1
    labels = []
    for idx in range(ds_size):
        label = ds[idx][label_idx]
        labels.append(label)
    # if use_fine_label:
    #     return np.array([info[2] for info in ds])
    # else:
    #     return np.array([info[1] for info in ds])
    return np.array(labels)
    
def sample_indices(indices,ratio):
    return np.random.choice(indices,int(ratio*len(indices)),replace=False)

def complimentary_mask(mask_length, active_spot):
    '''
    get an aversion of the active mask
    '''
    active_mask = np.zeros(mask_length, dtype=bool)
    active_mask[active_spot] = True
    advert_mask = ~active_mask 
    return advert_mask

def split_dataset(labels, label_summary, ratio=0 ):
    '''Label-wise split a dataset into two parts. in Part I: take every split_amt data-point per class.
    in part II: the rest of data
    
    Params: 
           labels: numpy array of subclass labels
           label_summary: what labels has data split
           split_ratio: how much to split into PART 1 from the dataset 
    Return:
           p1_indices: indices belonging to part 1 (split_ratio)
           p2_indices: indices belonging to part 2
    '''

    if torch.is_tensor(labels):
        labels = labels.numpy()
    ds_length = len(labels)
    p1_indices = []
    for c in label_summary:
        cls_indices = np.arange(ds_length)[labels == c]
        split_indices = sample_indices(cls_indices,ratio)
        p1_indices.append(split_indices)

    p1_indices = np.concatenate(p1_indices)
    mask_p2 = complimentary_mask(mask_length=ds_length,active_spot=p1_indices)
    p2_indices = np.arange(ds_length)[mask_p2]
    return p1_indices,p2_indices

def count_minority(ds):
    '''
    return #minority in ds
    '''
    minority_labels = [4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    cnt = 0
    ds_size = len(ds)
    for index in range(ds_size):
        if ds[index][2] in minority_labels:
            cnt += 1
    return cnt

def get_data_splits_list(epochs, select_fine_labels, label_map, ratio):
    data_split_env()
    ds_list = []
    for epo in range(epochs):
        ds = DataSplits(data_config['ds_root'],select_fine_labels,ratio)
        if select_fine_labels!=[] and (isinstance(label_map, dict)):
            ds.modify_coarse_label(label_map)
        ds_list.append(ds)
    return ds_list
