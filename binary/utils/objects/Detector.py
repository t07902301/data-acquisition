import sys
sys.path.append('..')
import numpy as np
from utils.detector.wrappers import SVMFitter, CLIPProcessor
from utils import config
import utils.objects.model as Model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import utils.objects.dataset as Dataset
from abc import abstractmethod
 
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

class Prototype():
    def __init__(self, data_transform:str) -> None:
        self.transform = data_transform
    @abstractmethod
    def fit(self, base_model:Model.prototype, data_loader):
        pass
    @abstractmethod
    def predict(self, data_loader, base_model:Model.prototype, compute_metrics=False):
        pass

class SVM(Prototype):
    def __init__(self, set_up_dataloader, clip_processor:CLIPProcessor, split_and_search=False, data_transform = 'clip') -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.fitter = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv=config['clf_args']['k-fold'], split_and_search = split_and_search)
        set_up_latent = get_latent(set_up_dataloader, clip_processor, self.transform)
        self.fitter.set_preprocess(set_up_latent) 
        # plt.hist(set_up_img[0], 50)
        # plt.savefig('figure/img.png')
        # plt.close()
    
    def fit(self, base_model:Model.prototype, data_loader):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        score = self.fitter.fit(latent, correctness)
        return score
        # fit_gts, fit_preds, _ = base_model.eval(fit_loader)
        # fit_latent = self.get_latent(fit_loader)
        # score = self.fitter.fit(model_preds=fit_preds, model_gts=fit_gts, latents=fit_latent, C=C)        
        # return score
    
    def predict(self, data_loader, base_model:Model.prototype, compute_metrics=False):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        _, dv, metric = self.fitter.predict(latent, correctness, compute_metrics)
        return dv, metric 

        # if compute_metrics:
        #     data_gts, dataset_preds, _ = base_model.eval(data_loader)
        # else:
        #     data_gts, dataset_preds = None, None
        # latent = self.get_latent(data_loader)
        # _, dv, metric = self.fitter.predict(latents=latent, model_gts=data_gts, compute_metrics=compute_metrics, model_preds=dataset_preds)
        # return dv, metric 

class resnet(Prototype):
    def __init__(self, num_class, use_pretrained=False, data_transform: str = None) -> None:
        super().__init__(data_transform)
        self.model = Model.resnet(num_class, use_pretrained)

    def fit(self, base_model:Model.prototype, data, data_loader, batch_size):
        split_labels = Dataset.data_config['train_label']
        train_indices, val_indices = Dataset.get_split_indices(Dataset.get_ds_labels(data), split_labels, 0.8)
        _, _, data_correctness = get_correctness(data_loader, base_model, self.transform)
        train_ds = torch.utils.data.Subset(data_correctness, train_indices)
        val_ds = torch.utils.data.Subset(data_correctness, val_indices)   
        print('Dstr CLF training : validation', len(train_ds), len(val_ds))
        generator = torch.Generator()
        generator.manual_seed(0)    
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)
        self.model.train(train_loader, val_loader)

    def predict(self, data_loader, base_model:Model.prototype, batch_size=16, compute_metrics=False):
        _, _, data_correctness = get_correctness(data_loader, base_model, self.transform)
        correctness_loader = torch.utils.data.DataLoader(data_correctness, batch_size = batch_size)
        gts, preds, confs = self.model.eval(correctness_loader)
        metrics = None
        if compute_metrics:
            metrics = balanced_accuracy_score(gts, preds) * 100
        return confs, metrics 

def load_clip(device):
    clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'], device=device)
    return clip_processor

import torch
def get_flattened(loader):
    img = []
    for x, y,fine_y in loader:
        img.append(torch.flatten(x, start_dim=1))
    return torch.cat(img, dim=0)

def get_latent(data_loader, clip_processor:CLIPProcessor = None, transform: str = None):
    if transform == 'clip':
        latent, _ = clip_processor.evaluate_clip_images(data_loader)  
    elif transform == 'flatten':
        latent = get_flattened(data_loader) 
    else:
        latent = loader2data(data_loader)
    return latent

def loader2data(loader):
    img = []
    for x, y,fine_y in loader:
        img.append(x)
    return torch.cat(img, dim=0)

def get_correctness(data_loader, model:Model.prototype, transform: str = None, clip_processor:CLIPProcessor = None):
    '''
    Return : \n 
    Data in Latent Space, Model Prediction Correctness of Data, Combined Latent Data and Model Correctness
    ''' 
    gts, preds, _ = model.eval(data_loader)
    data = get_latent(data_loader, clip_processor, transform)
    correctness_mask = (gts == preds)
    correctness = np.zeros(len(data), dtype = int)
    correctness[correctness_mask] = 1
    combined = []
    for idx in range(len(data)):
        combined.append((data[idx], correctness[idx], correctness[idx]))
    return data, correctness, combined 