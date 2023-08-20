import numpy as np
import pickle as pkl
import os

def complimentary_mask(mask_length, active_spot):
    '''
    reverse the active mask
    '''
    active_mask = np.zeros(mask_length, dtype=bool)
    active_mask[active_spot] = True
    advert_mask = ~active_mask 
    return advert_mask

def get_subset(dataset_labels, targets, dataset_indices, ratio=0):
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

def balanced_split(ratio, dataset:dict, categories, sessions):
    '''
    Split a Dataset Dict balanced from category, session to object. Ratio is to sample in object levels.
    '''

    obj_labels = np.asarray(dataset['object'])
    se_labels = np.asarray(dataset['session'])
    cat_labels = np.asarray(dataset['category'])
    data = np.asarray(dataset['data'])

    dataset_size = len(data)
    
    indices = np.arange(dataset_size)

    sampled_cat_indices = []

    for cat in categories:

        cat_indices = indices[cat_labels==cat]
        cat_session_labels = se_labels[cat_indices]

        objects = [i + cat * 5 for i in range(5)]

        sampled_session_indices = []
        for session in sessions: # session labels given a cat

            session_indices = cat_indices[cat_session_labels==session]
            session_obj_labels = obj_labels[session_indices]

            subset_indices = get_subset(session_obj_labels, objects, session_indices, ratio) # obj labels given a session
            sampled_session_indices.append(subset_indices)
        
        sampled_session_indices = np.concatenate(sampled_session_indices, axis = 0)

        sampled_cat_indices.append(sampled_session_indices)

    sampled_cat_indices = np.concatenate(sampled_cat_indices, axis = 0)

    return {
        'sampled': sampled_cat_indices,
        'others': complimentary_mask(dataset_size, sampled_cat_indices)
    }

def subset_dict(dataset, subset_indices):

    obj_labels = np.asarray(dataset['object'])
    se_labels = np.asarray(dataset['session'])
    cat_labels = np.asarray(dataset['category'])
    data = np.asarray(dataset['data'])
    
    return {
        'data': data[subset_indices],
        'session': se_labels[subset_indices],
        'object': obj_labels[subset_indices],
        'category': cat_labels[subset_indices]
    }

def get_meta_splits(data_config, meta_data):
    '''
    Get train, market, val, test set without any shift
    '''
    ratio = data_config['ratio']
    train_size = ratio["train_size"]
    test_size = ratio["test_size"]
    val_from_test = ratio['val_from_test']

    label_map = data_config['label_map']
    categories = list(label_map.keys()) if label_map != None else [i for i in range(10)]
    sessions = [i for i in range(11)]

    test_train_split = balanced_split(test_size, meta_data, categories, sessions)
    test_val_dict = subset_dict(meta_data, test_train_split['sampled'])
    train_market_dict = subset_dict(meta_data, test_train_split['others'])

    train_market_split = balanced_split(train_size, train_market_dict, categories, sessions)
    train_dict = subset_dict(train_market_dict, train_market_split['sampled'])
    market_dict = subset_dict(train_market_dict, train_market_split['others'])

    val_test_split = balanced_split(val_from_test, test_val_dict, categories, sessions)
    test_dict = subset_dict(test_val_dict, val_test_split['others'])
    val_dict = subset_dict(test_val_dict, val_test_split['sampled'])

    ds = {}
    ds['val'] =  val_dict
    ds['market'] =  market_dict
    ds['test'] = test_dict
    ds['train'] =  train_dict
    return ds
