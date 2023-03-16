import sys
sys.path.append('..')
from failure_directions.src.wrappers import SVMFitter,CLIPProcessor
import numpy as np
from utils.model import evaluate_model
from utils.dataset import complimentary_mask
import torch
from utils import config
def get_class_info(cls_label, ds_labels, ds_dv):
    '''
    get class information from a dataset
    '''
    cls_mask = ds_labels==cls_label
    ds_indices = np.arange(len(ds_labels))
    cls_indices = ds_indices[cls_mask]
    cls_dv = ds_dv[cls_mask] # cls_dv and cls_market_indices are mutually dependent, cls_market_indices index to locations of each image and its decision value
    return cls_indices, cls_mask, cls_dv
        
def sample_acquire(indices, sample_size):
    return np.random.choice(indices,sample_size,replace=False)
def dummy_acquire(cls_gt, cls_pred,method,img_num):
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

def apply_CLF(clf, model, data_loader, clip_features, ds_split):
    # Eval SVM
    ds_gt, ds_pred, ds_conf = evaluate_model(data_loader[ds_split],model)

    # Get decision values
    _, dv, _ = clf.predict(latents=clip_features[ds_split], compute_metrics=False, gts=ds_gt, preds=ds_pred)
    ds_info ={
        'dv': dv,
        'gt': ds_gt,
        'pred': ds_pred,
        'conf': ds_conf
    }
    return ds_info
        
def get_new_data_indices(new_model_config, ds_info, method, new_data_number_per_class):
    '''
    get new data from a dataset with info
    '''
    new_data_indices_total = []
    for c in range(new_model_config.class_number):
        cls_indices, cls_mask, cls_dv = get_class_info(c, ds_info['gt'], ds_info['dv'])
        sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
        if (method == 'hard') or (method=='easy'):
            cls_new_data_indices = dummy_acquire(cls_gt=ds_info['gt'][cls_mask],cls_pred=ds_info['pred'][cls_mask],method=method, img_num=new_data_number_per_class)
            new_data_indices = cls_indices[cls_new_data_indices]
            # visualize_images(class_imgs,new_data_indices,cls_dv[new_data_indices],path=os.path.join('vis',method))
        else:
            if method == 'dv':
                new_data_indices = get_top_values(cls_indices[sorted_idx],new_data_number_per_class)
            elif method == 'sm':
                new_data_indices = sample_acquire(cls_indices,new_data_number_per_class)
            elif method == 'mix':
                dv_results = get_top_values(cls_indices[sorted_idx],new_data_number_per_class-new_data_number_per_class//2)
                sample_results = sample_acquire(cls_indices,new_data_number_per_class//2)
                new_data_indices = np.concatenate((dv_results,sample_results)) 
            elif method == 'conf':
                masked_conf = ds_info['conf'][cls_mask]
                conf_sorted_idx = np.argsort(masked_conf) 
                new_data_indices = get_top_values(cls_indices[conf_sorted_idx],new_data_number_per_class)
        new_data_indices_total.append(new_data_indices) 
    return np.concatenate(new_data_indices_total)   
        
def get_top_values(sorted_indices, K=0, clf='SVM'):
    '''
    return indices of images with top decision scores
    '''
    if clf == 'SVM':
        dv_indices = sorted_indices[:K]
    else:
        dv_indices = sorted_indices[::-1][:K] # decision scores from model confidence is non-negative
    return dv_indices

def get_CLF(base_model,dataloaders, svm_fit_label= 'val'):
    clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'])
    svm_fit_gt,svm_fit_pred,_ = evaluate_model(dataloaders[svm_fit_label],base_model) # gts, preds, loss
    clip_features = {}
    for split, loader in dataloaders.items():
        if (split == 'train') or (split=='market_aug'):
            continue
        clip_features[split] = clip_processor.evaluate_clip_images(loader)
    svm_fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv= config['clf_args']['k-fold'])
    svm_fitter.set_preprocess(clip_features['train_clip'])
    score = svm_fitter.fit(preds=svm_fit_pred, gts=svm_fit_gt, latents=clip_features[svm_fit_label])

    return svm_fitter,clip_features,score

