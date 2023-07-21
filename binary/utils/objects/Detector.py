import sys
sys.path.append('..')
import numpy as np
import utils.detector.wrappers as wrappers
from utils import config
import utils.objects.model as Model
import utils.objects.dataset as Dataset
from abc import abstractmethod
 
def statistics(values, stat_type, n_data_list = None):
    '''
    pretrained model | seq_clf\n
    score/precision: (epoch, value)
    '''
    values = np.array(values)
    if n_data_list is None:
        print('Distribution Classifier Average Precision {}: {}%'.format(stat_type, np.round(np.mean(values), decimals=3).tolist()))
    else: 
        for idx, n_data in enumerate(n_data_list):
            print('#new data:', n_data)       
            print('Distribution Classifier Average Precision {}: {}%'.format(stat_type, np.round(np.mean(values[:,idx]), decimals=3).tolist()))

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
        self.model = wrappers.Prototype(None, None)
    @abstractmethod
    def fit(self, base_model:Model.prototype, data_loader, data=None, batch_size = None):
        pass
    @abstractmethod
    def predict(self, data_loader, base_model:Model.prototype, compute_metrics=False):
        '''
        Decision Scores + Performance Metrics
        '''
        pass

    def save(self, path):
        self.model.export(path)
        print('{} log save to {}'.format('Detector', path))

    def load(self, path):
        self.model.import_model(path)
        print('{} log load from {}'.format('Detector', path))
        
class SVM(Prototype):
    def __init__(self, clip_processor:wrappers.CLIPProcessor, split_and_search=False, data_transform = 'clip') -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.model = wrappers.SVM(args=config['clf_args'], cv=config['clf_args']['k-fold'], split_and_search = split_and_search, do_normalize=True, do_standardize=False)
        # #TODO take the mean and std assumed norm dstr
        # set_up_latent = get_latent(set_up_dataloader, clip_processor, self.transform)
        # self.model.set_preprocess(set_up_latent) 
    
    def fit(self, base_model:Model.prototype, data_loader, data=None, batch_size=None):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        self.model.set_preprocess(latent) #TODO val set norm stat may be accurate as those from train_clip
        score = self.model.fit(latent, correctness)
        return score
    
    def predict(self, data_loader, base_model:Model.prototype, compute_metrics=False):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        _, dv, metric = self.model.predict(latent, correctness, compute_metrics)
        return dv, metric 

class LogRegressor(Prototype):
    def __init__(self, data_transform: str, clip_processor:wrappers.CLIPProcessor) -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.model = wrappers.LogRegressor(do_normalize=True, do_standardize=False)

    def fit(self, base_model:Model.prototype, data_loader, data=None, batch_size = None):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        self.model.set_preprocess(latent) #TODO val set norm stat may be accurate as those from train_clip
        self.model.fit(latent, correctness)
        
    def predict(self, data_loader, base_model: Model.prototype, compute_metrics=False):
        latent = get_latent(data_loader, self.clip_processor, self.transform)
        gts = None
        if compute_metrics:
            _, gts, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        dv, metrics = self.model.predict(latent, gts, compute_metrics)
        return dv, metrics     
    
def factory(detector_type, clip_processor:wrappers.CLIPProcessor, split_and_search=False, data_transform = 'clip'):
    if detector_type == 'svm':
        return SVM(clip_processor, split_and_search, data_transform)
    elif detector_type == 'resnet':
        return resnet()
    elif detector_type == 'logregs':
        return LogRegressor(data_transform, clip_processor)
    else:
        return RandomForest(data_transform, clip_processor)

def load_clip(device):
    clip_processor = wrappers.CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'], device=device)
    return clip_processor

import torch
def get_flattened(loader):
    img = []
    for batch_info in loader:
        img.append(torch.flatten(batch_info[0], start_dim=1))
    return torch.cat(img, dim=0)

def get_latent(data_loader, clip_processor:wrappers.CLIPProcessor = None, transform: str = None):
    if transform == 'clip':
        latent, _ = clip_processor.evaluate_clip_images(data_loader)  
    elif transform == 'flatten':
        latent = get_flattened(data_loader) 
    else:
        latent = loader2data(data_loader)
    return latent

def loader2data(loader):
    img = []
    for batch_info in loader:
        img.append(batch_info[0])
    return torch.cat(img, dim=0)

def get_correctness(data_loader, model:Model.prototype, transform: str = None, clip_processor:wrappers.CLIPProcessor = None):
    '''
    Return : \n 
    Data in Latent Space, Base Model Prediction Correctness of Data, Combined Latent Data and Model Correctness
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
    
class resnet(Prototype):
    def __init__(self, data_transform: str = None) -> None:
        super().__init__(data_transform)
        self.model = Model.resnet(num_class=2, use_pretrained=False)

    def fit(self, base_model:Model.prototype, data_loader, data=None, batch_size = None):
        split_labels = Dataset.data_config['train_label']
        train_indices, val_indices = Dataset.get_split_indices(Dataset.get_ds_labels(data), split_labels, 0.8)
        _, _, data_correctness = get_correctness(data_loader, base_model, self.transform)
        train_ds = torch.utils.data.Subset(data_correctness, train_indices)
        val_ds = torch.utils.data.Subset(data_correctness, val_indices)   
        print('Dstr CLF - training : validation =', len(train_ds), len(val_ds))
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
        # if compute_metrics:
        #     metrics = balanced_accuracy_score(gts, preds) * 100
        return confs, metrics 

from sklearn.ensemble import RandomForestClassifier
class RandomForest(Prototype):
    def __init__(self, data_transform: str, clip_processor:wrappers.CLIPProcessor) -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.model = RandomForestClassifier(n_estimators=5)
    def fit(self, base_model: Model.prototype, data_loader, data=None, batch_size=None):
        latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        self.model = self.model.fit(latent, correctness)
    def predict(self, data_loader, base_model: Model.prototype, compute_metrics=False):
        data = get_latent(data_loader, self.clip_processor, self.transform)
        preds = self.model.predict(data)
        dataset = Dataset.loader2dataset(data_loader)
        gt_labels = Dataset.get_ds_labels(dataset)
        # TODO 
        confs = self.model.predict_proba(data)
        cls_conf = []
        for idx in (range(len(confs))):
            cls_conf.append(confs[idx][gt_labels[idx]])
        metrics = None
        # if compute_metrics:
        #     _, gts, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
        #     metrics = balanced_accuracy_score(gts, preds) * 100
        return np.array(cls_conf), metrics  
