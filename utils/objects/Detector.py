import numpy as np
import utils.detector.wrappers as wrappers
import utils.objects.model as Model
import utils.objects.dataloader as dataloader_utils
from utils.logging import *
import utils.detector.weakness as weakness_utils

class Prototype():
    def __init__(self, data_transform:str) -> None:
        self.transform = data_transform
        self.model = wrappers.Prototype(None, None)
    
    def fit(self, base_model:Model.Prototype, data_loader, weaness_label_generator):
        '''
        Train Detector and Log out the Best Cross Validation Performance for Reguarization Penalty
        '''
        latent, latent_gts = dataloader_utils.get_latent(data_loader, self.clip_processor, self.transform)
        # correctness = get_correctness(data_loader, base_model, latent_gts)
        self.weaness_label_generator = weakness_utils.factory(weaness_label_generator, base_model, data_loader)
        non_weakness = self.weaness_label_generator.run(data_loader, base_model, latent_gts)
        logger.info('val shift acc told by Detector:{}'.format(non_weakness.mean()))

        self.model.set_preprocess(latent) 
        score = self.model.fit(latent, non_weakness)
        logger.info('Best CV Score: {}'.format(score))
        
    def predict(self, data_loader, base_model:Model.Prototype=None, metrics=None):
        '''
        Feature Scores + Performance Metrics
        '''
        latent, latent_gts = dataloader_utils.get_latent(data_loader, self.clip_processor, self.transform)
        if metrics != None:
            # correctness = get_correctness(data_loader, base_model, latent_gts)
            non_weakness = self.weaness_label_generator.run(data_loader, base_model, latent_gts)
            weakness_score, metric = self.model.predict(latent, non_weakness, metrics)
        else:
            weakness_score, _ = self.model.predict(latent)
            metric = None
        return weakness_score, metric 

    def save(self, path):
        self.model.export(path)
        logger.info('{} log save to {}'.format('Detector', path))

    def load(self, path):
        self.model.import_model(path)
        logger.info('{} log load from {}'.format('Detector', path))
        
class SVM(Prototype):
    def __init__(self, config, clip_processor:wrappers.CLIPProcessor, split_and_search=True, data_transform = 'clip') -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.model = wrappers.SVM(args=config['detector_args'], cv=config['detector_args']['k-fold'], split_and_search = split_and_search)
        # #TODO take the mean and std assumed norm dstr
        # set_up_latent = get_latent(set_up_dataloader, clip_processor, self.transform)
        # self.model.set_preprocess(set_up_latent) 

class LogRegressor(Prototype):
    def __init__(self, config, clip_processor:wrappers.CLIPProcessor, split_and_search=True, data_transform = 'clip') -> None:
        super().__init__(data_transform)
        self.clip_processor = clip_processor
        self.model = wrappers.LogRegressor(cv=config['detector_args']['k-fold'], split_and_search=split_and_search)
    
def factory(detector_type, config, clip_processor:wrappers.CLIPProcessor, split_and_search=True, data_transform = 'clip'):
    if detector_type == 'svm':
        return SVM(config, clip_processor, split_and_search, data_transform)
    elif detector_type == 'logregs':
        return LogRegressor(config, clip_processor, split_and_search, data_transform)
    else:
        logger.info('Weakness Detector is not Implemented.')
        exit()
    # elif detector_type == 'resnet':
    #     return resnet()
    # else:
    #     return RandomForest(data_transform, clip_processor)

def load_clip(device, mean, std):
    clip_processor = wrappers.CLIPProcessor(ds_mean=mean, ds_std=std, device=device)
    return clip_processor

def get_correctness(data_loader, model:Model.Prototype, loader_gts):
    '''
    Base Model Prediction Correctness as True Labels for Detectors
    ''' 
    gts, preds, _ = model.eval(data_loader)
    assert (gts != loader_gts).sum() == 0, 'Train Loader Shuffles!: {}'.format((gts != loader_gts).sum())
    correctness_mask = (gts == preds)
    logger.info('Model Acc in Detector Traning Data: {}'.format(correctness_mask.mean()))
    data_loader_size = dataloader_utils.get_size(data_loader)
    correctness = np.zeros(data_loader_size, dtype = int)
    correctness[correctness_mask] = 1
    return correctness
    
# def combine_latent_correctness(latent, correctness):
#     combined = []
#     for idx in range(len(latent)):
#         combined.append((latent[idx], correctness[idx]))
#     return combined

# class resnet(Prototype):
#     def __init__(self, data_transform: str = None) -> None:
#         super().__init__(data_transform)
#         self.model = Model.resnet(num_class=2, use_pretrained=False)

#     def fit(self, base_model:Model.Prototype, data_loader, data=None, batch_size = None):
#         split_labels = Dataset.data_config['train_label']
#         train_indices, val_indices = Dataset.get_split_indices(Dataset.get_ds_labels(data), split_labels, 0.8)
#         _, _, data_correctness = get_correctness(data_loader, base_model, self.transform)
#         train_ds = torch.utils.data.Subset(data_correctness, train_indices)
#         val_ds = torch.utils.data.Subset(data_correctness, val_indices)   
#         logger.info('Dstr CLF - training : validation =', len(train_ds), len(val_ds))
#         generator = torch.Generator()
#         generator.manual_seed(0)    
#         train_loader = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True, drop_last=True)
#         val_loader = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)
#         self.model.train(train_loader, val_loader)

#     def predict(self, data_loader, base_model:Model.Prototype, batch_size=16, compute_metrics=False):
#         _, _, data_correctness = get_correctness(data_loader, base_model, self.transform)
#         correctness_loader = torch.utils.data.DataLoader(data_correctness, batch_size = batch_size)
#         gts, preds, confs = self.model.eval(correctness_loader)
#         metrics = None
#         # if compute_metrics:
#         #     metrics = balanced_accuracy_score(gts, preds) * 100
#         return confs, metrics 

# from sklearn.ensemble import RandomForestClassifier
# class RandomForest(Prototype):
#     def __init__(self, data_transform: str, clip_processor:wrappers.CLIPProcessor) -> None:
#         super().__init__(data_transform)
#         self.clip_processor = clip_processor
#         self.model = RandomForestClassifier(n_estimators=5)
#     def fit(self, base_model: Model.Prototype, data_loader, data=None, batch_size=None):
#         latent, correctness, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
#         self.model = self.model.fit(latent, correctness)
#     def predict(self, data_loader, base_model: Model.Prototype, compute_metrics=False):
#         data = dataloader_utils.get_latent(data_loader, self.clip_processor, self.transform)
#         preds = self.model.predict(data)
#         dataset = Dataset.loader2dataset(data_loader)
#         gt_labels = Dataset.get_ds_labels(dataset)
#         # TODO 
#         confs = self.model.predict_proba(data)
#         cls_conf = []
#         for idx in (range(len(confs))):
#             cls_conf.append(confs[idx][gt_labels[idx]])
#         metrics = None
#         # if compute_metrics:
#         #     _, gts, _ = get_correctness(data_loader, base_model, self.transform, self.clip_processor)
#         #     metrics = balanced_accuracy_score(gts, preds) * 100
#         return np.array(cls_conf), metrics  
