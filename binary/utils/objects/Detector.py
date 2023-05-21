import sys
sys.path.append('..')
import numpy as np
from utils.detector.wrappers import SVMFitter, CLIPProcessor
from utils import config
import utils.objects.model as Model

def statistics(values, stat_type, n_data_list = None):
    '''
    pretrained model | seq_clf\n
    score/precision: (epoch, value)
    '''
    values = np.array(values)
    if n_data_list is None:
        print('OOD detector average {}: {}%'.format(stat_type, np.round(np.mean(values), decimals=3).tolist()))
    else: 
        for idx, n_data in enumerate(n_data_list):
            print('#new data:', n_data)       
            print('OOD detector average {}: {}%'.format(stat_type, np.round(np.mean(values[:,idx]), decimals=3).tolist()))

def precision(clf, clip_processor, dataloader, base_model):
    data_clip = clip_processor.evaluate_clip_images(dataloader)
    gt, pred, conf = Model.evaluate(dataloader, base_model)
    _, dv, _ = clf.predict(latents=data_clip, gts=gt, compute_metrics=False, preds=None)
    dataset_len = len(gt)
    clf_cls_incorrect = np.arange(dataset_len)[dv<=0]
    real_cls_incorrect = np.arange(dataset_len)[gt!=pred]
    return np.intersect1d(clf_cls_incorrect, real_cls_incorrect).size / clf_cls_incorrect.size

class SVM():
    def __init__(self, set_up_dataloader, clip_processor:CLIPProcessor, split_and_search=False) -> None:
        self.clip_processor = clip_processor
        set_up_embedding, _ = self.clip_processor.evaluate_clip_images(set_up_dataloader)        
        self.fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv=config['clf_args']['k-fold'], split_and_search = split_and_search)
        self.fitter.set_preprocess(set_up_embedding) 

    def fit(self, base_model:Model.prototype, fit_data):
        fit_gts, fit_preds, _ = base_model.eval(fit_data)
        fit_embedding, fit_gts_clip = self.clip_processor.evaluate_clip_images(fit_data)
        assert (fit_gts_clip == fit_gts).sum() == len(fit_gts)
        score = self.fitter.fit(model_preds=fit_preds, model_gts=fit_gts, latents=fit_embedding)        
        return score
    
    def predict(self, data_loader, compute_metrics=False, base_model:Model.prototype=None):
        if compute_metrics:
            _, dataset_preds, _ = base_model.eval(data_loader)
        else:
            dataset_preds = None
        embedding, data_gts = self.clip_processor.evaluate_clip_images(data_loader)
        _, dv, precision = self.fitter.predict(latents=embedding, gts=data_gts, compute_metrics=compute_metrics, preds=dataset_preds)
        return dv, precision        
    
def load_clip(device):
    clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'], device=device)
    return clip_processor