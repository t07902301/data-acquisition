import numpy as np
import torch
def get_loader_labels(dataloader):
    gts = []
    for x, y,fine_y in dataloader:
        gts.append(y)
    return torch.cat(gts).numpy()

def extract_class_indices(cls_label, ds_labels):
    '''
    Get indices of a class from dataset with 'ds_labels'
    '''
    cls_mask = ds_labels==cls_label
    ds_indices = np.arange(len(ds_labels))
    cls_indices = ds_indices[cls_mask]
    return cls_indices
        
def sample_acquire(indices, sample_size):
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

def get_top_values(sorted_indices, K=0, clf='SVM'):
    '''
    return indices of images with top decision scores
    '''
    if clf == 'SVM':
        dv_indices = sorted_indices[:K]
    else:
        dv_indices = sorted_indices[::-1][:K] # decision scores from clf confidence is non-negative
    return dv_indices
