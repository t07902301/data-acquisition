import numpy as np

def extract_class_indices(cls_label, ds_labels):
    '''
    get class (cls_label) information from a dataset
    '''
    cls_mask = ds_labels==cls_label
    ds_indices = np.arange(len(ds_labels))
    cls_indices = ds_indices[cls_mask]
    return cls_indices
        
def sample(group, sample_size):
    '''
    Sample members of a given size from a group; \n 
    Return an entire group if expected sample is larger than group itself.
    '''
    if len(group) <= sample_size:
        return group
    else:
        return np.random.choice(group,sample_size,replace=False)

def dummy_acquire(cls_gt, cls_pred, method, img_num):
    if method == 'hard':
        result_mask = cls_gt != cls_pred
    else:
        result_mask = cls_gt == cls_pred
    result_mask_indices = np.arange(len(result_mask))[result_mask]
    if result_mask.sum() > img_num:
        new_img_indices = sample(result_mask_indices,img_num)
    else:
        print('class result_mask_indices with a length',len(result_mask_indices))
        new_img_indices = result_mask_indices
    return new_img_indices  

def get_top_values_indices(values, K=0, order='descend'):
    '''
    return indices of top values by indicated order
    '''
    if order == 'ascend':
        sorting_idx = np.argsort(values) # ascending order
    else:
        sorting_idx = np.argsort(-values) # descending order
    top_val_indices = sorting_idx[:K]
    return top_val_indices

def get_threshold_indices(values, threshold, anchor_threshold=0.7):
    threshold_mask = (values >= threshold)
    value_indices = np.arange(len(values))
    threshold_value_indicies = value_indices[threshold_mask]
    sample_size = (values >= anchor_threshold).sum()
    sample_indices = sample(threshold_value_indicies, sample_size)
    return sample_indices
        
def get_threshold_range(values, lower_t):
    anchor_size = ((values>0.8)).sum()
    upper_t = lower_t + 0.2
    lower_mask = (values >= lower_t)
    upper_mask = (values <= upper_t)
    selected_mask = (lower_mask == upper_mask)

    # lower_value_indices = np.arange(len(values))[lower_mask]
    # lower_value = values[lower_value_indices]
    # upper_mask = (lower_value >= upper_t)
    # upper_value_indices = np.arange(len(lower_value))[upper_mask]
    # upper_value = lower_value[upper_value_indices]


def get_top_indices_threshold(values, K, threshold, order='descend'):
    '''
    return indices by indicated threshold
    '''
    threshold_mask = (values >= threshold)
    threshold_value = values[threshold_mask]
    value_indices = np.arange(len(values))
    threshold_value_indicies = value_indices[threshold_mask]
    top_indicies = get_top_values_indices(threshold_value, K, order)
    return threshold_value_indicies[top_indicies]

def get_gt_probab(gts, probab):
    '''
    Return Prediction Probability of True Labels \n
    probab: (n_samples, n_class)
    '''
    return probab[np.arange(len(gts)), gts]

def get_gt_distance(gts, decision_values):
    '''
    Return Distance to HyperPlane of True Labels \n
    decision_values: (n_samples)
    '''
    cls_0_mask = (gts==0)
    cls_1_mask = ~cls_0_mask
    distance = np.zeros(len(gts))
    distance[cls_0_mask] = (0 - decision_values[cls_0_mask])
    distance[cls_1_mask] = (decision_values[cls_1_mask])
    return distance[gts]