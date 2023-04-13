import sys
sys.path.append('..')
import numpy as np
from failure_directions.src.wrappers import SVMFitter,CLIPProcessor
from utils.env import clip_env
from utils import config
from utils.acquistion import get_loader_labels
import utils.objects.model as Model

def statistics(epochs, score, incor_precision, n_data_list = None):
    if n_data_list is None:
        cv_score= []
        cv_precision = []
        for epo in range(epochs):
            cv_score.append(score[epo])
            cv_precision.append(incor_precision[epo])
        print('average CV score:', np.mean(cv_score, axis=0))
        print('average precision:', np.mean(incor_precision, axis=0))
    else:
        for idx, n_data in enumerate(n_data_list):
            cv_score= []
            cv_precision = []
            for epo in range(epochs):
                cv_score.append(score[epo][idx])
                cv_precision.append(incor_precision[epo][idx])
            print('n_data:', n_data)       
            print('average CV score:', np.mean(cv_score, axis=0))
            print('average precision:', np.mean(incor_precision, axis=0))

def precision(clf, clip_processor, dataloader, base_model):
    data_clip = clip_processor.evaluate_clip_images(dataloader)
    gt, pred, conf = Model.evaluate(dataloader, base_model)
    _, dv, _ = clf.predict(latents=data_clip, gts=gt, compute_metrics=False, preds=None)
    dataset_len = len(gt)
    clf_cls_incorrect = np.arange(dataset_len)[dv<=0]
    real_cls_incorrect = np.arange(dataset_len)[gt!=pred]
    return np.intersect1d(clf_cls_incorrect, real_cls_incorrect).size / clf_cls_incorrect.size

def get_CLF(base_model, dataloaders, svm_fit_label= 'val'):
    clip_env()
    clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'])
    clip_set_up = clip_processor.evaluate_clip_images(dataloaders['train_clip'])
    svm_fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv= config['clf_args']['k-fold'])
    svm_fitter.set_preprocess(clip_set_up)
    svm_fit_gt,svm_fit_pred,_ = Model.evaluate(dataloaders[svm_fit_label],base_model) # gts, preds, loss
    clip_fit = clip_processor.evaluate_clip_images(dataloaders[svm_fit_label])
    score = svm_fitter.fit(preds=svm_fit_pred, gts=svm_fit_gt, latents=clip_fit)
    return svm_fitter, clip_processor, score


def apply_CLF(clf, data_loader, clip_processor):
    '''
    Get DV from CLF, gt from data
    '''
    # ds_gt, ds_pred, ds_conf = Model.evaluate(data_loader,model)
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
