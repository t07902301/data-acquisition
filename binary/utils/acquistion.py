import numpy as np
import torch
def get_loader_labels(dataloader):
    gts = []
    for x, y,fine_y in dataloader:
        gts.append(y)
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

def get_top_values_indices(values, K=0, clf='SVM'):
    '''
    return indices of images with top decision scores
    '''
    sorting_idx = np.argsort(values)
    value_indices = np.arange(len(values))
    sorted_val_indices = value_indices[sorting_idx]
    if clf == 'SVM':
        top_val_indices = sorted_val_indices[:K]
    else:
        top_val_indices = sorted_val_indices[::-1][:K] # decision scores from clf confidence is non-negative
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
    