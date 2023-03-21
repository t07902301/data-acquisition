import sys
sys.path.append('..')
from failure_directions.src.wrappers import SVMFitter,CLIPProcessor
from utils.model import evaluate_model
from utils import *
from utils.Config import *
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

def apply_CLF(clf, data_loader, clip_processor, model):
    ds_gt, ds_pred, ds_conf = evaluate_model(data_loader,model)
    ds_clip = clip_processor.evaluate_clip_images(data_loader)
    _, dv, _ = clf.predict(latents=ds_clip, compute_metrics=False, gts=ds_gt, preds=ds_pred)
    ds_info ={
        'dv': dv,
        'gt': ds_gt,
        'pred': ds_pred,
        'conf': ds_conf
    }
    return ds_info

def get_new_data_indices(method,new_data_number_per_class,class_number, ds_info):
    '''
    get new data from a dataset with info
    '''
    new_data_indices_total = []
    # cls_indices_total = []
    # cls_conf_total = []
    for c in range(class_number):
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
                cls_conf = ds_info['conf'][cls_mask]
                conf_sorted_idx = np.argsort(cls_conf) 
                new_data_indices = get_top_values(cls_indices[conf_sorted_idx],new_data_number_per_class)
        new_data_indices_total.append(new_data_indices) 
        # cls_indices_total.append(cls_indices)
        # cls_conf_total.append(cls_conf)
    # torch.save(cls_indices_total,'new_model/distribution/3-class-mini-small/64/retrain/non-pure/log/conf/indx_per_class.pt')
    # torch.save(new_data_indices_total,'new_model/distribution/3-class-mini-small/64/retrain/non-pure/log/conf/selected_indx_per_class.pt')
    # torch.save(cls_conf_total, 'new_model/distribution/3-class-mini-small/64/retrain/non-pure/log/conf/conf_per_class.pt')
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
    clip_set_up = clip_processor.evaluate_clip_images(dataloaders['train_clip'])
    svm_fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv= config['clf_args']['k-fold'])
    svm_fitter.set_preprocess(clip_set_up)
    svm_fit_gt,svm_fit_pred,_ = evaluate_model(dataloaders[svm_fit_label],base_model) # gts, preds, loss
    clip_svm_fit = clip_processor.evaluate_clip_images(dataloaders[svm_fit_label])
    score = svm_fitter.fit(preds=svm_fit_pred, gts=svm_fit_gt, latents=clip_svm_fit)
    return svm_fitter, clip_processor, score

def log_data(data, model_cofig:NewModelConfig, acquisition_config):
    log_config = LogConfig(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir,pure=model_cofig.pure,setter=model_cofig.setter)
    log_config.set_path(acquisition_config)
    torch.save(data, log_config.path)
