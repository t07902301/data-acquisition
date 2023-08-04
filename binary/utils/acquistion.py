import numpy as np
import torch
def get_loader_labels(dataloader):
    gts = []
    for batch_info in dataloader:
        gts.append(batch_info[1])
    return torch.cat(gts).numpy()

def extract_class_indices(cls_label, ds_labels):
    '''
    get class (cls_label) information from a dataset
    '''
    cls_mask = ds_labels==cls_label
    ds_indices = np.arange(len(ds_labels))
    cls_indices = ds_indices[cls_mask]
    return cls_indices
        
def sample_acquire(values, sample_size):
    indices = np.arange(len(values))
    return np.random.choice(indices,sample_size,replace=False)

def dummy_acquire(cls_gt, cls_pred, method, img_num):
    if method == 'hard':
        result_mask = cls_gt != cls_pred
    else:
        result_mask = cls_gt == cls_pred
    result_mask_indices = np.arange(len(result_mask))[result_mask]
    if result_mask.sum() > img_num:
        new_img_indices = sample_acquire(result_mask_indices,img_num)
    else:
        print('class result_mask_indices with a length',len(result_mask_indices))
        new_img_indices = result_mask_indices
    return new_img_indices  

def get_top_values_indices(values, K=0):
    '''
    return indices with top (target) decision scores from an ascending order
    '''
    sorting_idx = np.argsort(values) # ascending order
    value_indices = np.arange(len(values))
    sorted_val_indices = value_indices[sorting_idx]
    top_val_indices = sorted_val_indices[:K]
    return top_val_indices

def get_in_bound_top_indices(values, K, bound):
    # Get in_bound values and their indices
    in_bound_mask = values <= bound
    in_bound_val = values[in_bound_mask]
    val_indices = np.arange(len(values))
    in_bound_val_indicies = val_indices[in_bound_mask]
    # Get top indices from sorted array
    sorting_idx = np.argsort(in_bound_val)
    sorted_in_bound_val_indices = in_bound_val_indicies[sorting_idx]
    top_idx = sorted_in_bound_val_indices[:K]
    return top_idx
    
def get_probab_diff(gts, probab):
    cls_0_mask = (gts==0)
    cls_1_mask = ~cls_0_mask
    probab_diff = np.zeros(len(gts))
    probab_diff[cls_0_mask] = (0.5 - probab[cls_0_mask])
    probab_diff[cls_1_mask] = (probab[cls_1_mask] - 0.5)
    return probab_diff

def get_probab_gts(gts, probab):
    return probab[np.arange(len(gts)), gts]

def get_distance_diff(gts, distance):
    cls_0_mask = (gts==0)
    cls_1_mask = ~cls_0_mask
    probab_diff = np.zeros(len(gts))
    probab_diff[cls_0_mask] = (0 - distance[cls_0_mask])
    probab_diff[cls_1_mask] = (distance[cls_1_mask] - 0)
    return probab_diff