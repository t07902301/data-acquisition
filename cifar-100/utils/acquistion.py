import sys
sys.path.append('..')
from failure_directions.src.wrappers import SVMFitter,CLIPProcessor
from utils.model import evaluate_model
from utils.Config import *
from utils.env import clip_env
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

def apply_CLF(clf, data_loader, clip_processor):
    # ds_gt, ds_pred, ds_conf = evaluate_model(data_loader,model)
    ds_gts = get_loader_labels(data_loader)
    ds_clip = clip_processor.evaluate_clip_images(data_loader)
    _, dv, _ = clf.predict(latents=ds_clip, gts=ds_gts, compute_metrics=False, preds=None)
    ds_info ={
        'dv': dv,
        'gt': ds_gts,
        # 'pred': ds_pred,
        # 'conf': ds_conf
    }
    return ds_info

def get_top_values(sorted_indices, K=0, clf='SVM'):
    '''
    return indices of images with top decision scores
    '''
    if clf == 'SVM':
        dv_indices = sorted_indices[:K]
    else:
        dv_indices = sorted_indices[::-1][:K] # decision scores from clf confidence is non-negative
    return dv_indices

def get_CLF(base_model, dataloaders, svm_fit_label= 'val'):
    clip_env()
    clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'])
    clip_set_up = clip_processor.evaluate_clip_images(dataloaders['train_clip'])
    svm_fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv= config['clf_args']['k-fold'])
    svm_fitter.set_preprocess(clip_set_up)
    svm_fit_gt,svm_fit_pred,_ = evaluate_model(dataloaders[svm_fit_label],base_model) # gts, preds, loss
    clip_svm_fit = clip_processor.evaluate_clip_images(dataloaders[svm_fit_label])
    score = svm_fitter.fit(preds=svm_fit_pred, gts=svm_fit_gt, latents=clip_svm_fit)
    return svm_fitter, clip_processor, score