import utils.objects.model as Model
import utils.objects.Config as Config
from utils.objects.log import Log
import utils.dataset.wrappers as Dataset
import utils.objects.Detector as Detector
import numpy as np
import torch
from utils.logging import *

class Cifar():
    def __init__(self) -> None:
        pass

    def label_stat(self, dataset, checked_labels):
        ds_labels = Dataset.Cifar().get_labels(dataset, use_fine_label=True)
        check_labels_cnt = 0
        for label in checked_labels:
            check_labels_cnt += (label==ds_labels).sum()
        return check_labels_cnt
    
class Core():
    def __init__(self) -> None:
        pass

    def option_label_stat(self, dataset, checked_labels, option):
        check_labels_cnt = 0
        check_labels_indices_mask = np.zeros(len(dataset), dtype=bool)

        ds_labels = Dataset.Core().get_labels(dataset, option)
        for label in checked_labels:
            checked_mask = (label==ds_labels)
            check_labels_cnt += (checked_mask).sum()
            check_labels_indices_mask[checked_mask] = True

        check_labels_indices = np.arange(len(dataset))[check_labels_indices_mask]
        
        return check_labels_cnt, check_labels_indices

    def label_stat(self, dataset, remove_config, option):
        check_labels_cnt = 0
        check_labels_indices = []
        if option == 'both':
            for op in ['session', 'object']:
                checked_labels = remove_config[op]
                _, option_indices = self.option_label_stat(dataset, checked_labels, op)
                check_labels_indices.append(option_indices)
            return len(set(np.concatenate(check_labels_indices)))
        else:
            checked_labels = remove_config[option]
            check_labels_cnt, _ = self.option_label_stat(dataset, checked_labels, option)
            return check_labels_cnt

class Info():
    def __init__(self, dataset_splits: Dataset.DataSplits, name, new_batch_size) -> None:
        assert name != 'train', 'Avoid dataloader shuffles!'
        self.batch_size = new_batch_size
        self.dataset = dataset_splits.dataset[name]
        self.loader = dataset_splits.loader[name]

def get_new_data_max_weakness_score(clf:Detector.SVM , acquisition_config:Config.Acquisition, model_config:Config.NewModel, data_splits:Dataset.DataSplits):
    if 'seq' in acquisition_config.method: 
        log = Log(model_config, 'clf')
        clf = log.import_log(acquisition_config)    
    log = Log(model_config, 'indices')
    new_data_indices = log.import_log(acquisition_config)
    new_data = torch.utils.data.Subset(data_splits.dataset['market'], new_data_indices)    
    train_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                            num_workers= Dataset.n_workers)
    train_weakness_score, _ = clf.predict(train_data_loader)
    return np.max(train_weakness_score)

def error_label_stat(split_name, data_split:Dataset.DataSplits, model:Model.Prototype, remove_config, option):
    '''
    Get target labels proportion on errors
    '''
    gt,pred,_  = model.eval(data_split.loader[split_name])
    dataset = data_split.dataset[split_name]
    dataset_idx = np.arange(len(dataset))
    error_mask = (gt!=pred)
    error_idx = dataset_idx[error_mask]
    error_dataset = torch.utils.data.Subset(dataset,error_idx)
    if data_split.dataset_name == 'core':
        return Core().label_stat(error_dataset, remove_config, option) / len(error_dataset) * 100
    else:
        return Cifar().label_stat(error_dataset, remove_config) / len(error_dataset) * 100

def evaluation_metric(dataloader, old_model:Model.Prototype, ensemble_decision=None, new_model:Model.Prototype=None):
    gt, pred,_  = old_model.eval(dataloader)
    indices = np.arange(len(gt))
    old_correct_mask = (gt == pred)
    old_incorrect_mask = ~old_correct_mask
    old_correct_indices = indices[old_correct_mask]
    old_incorrect_indices = indices[old_incorrect_mask]

    if ensemble_decision is not None:
        new_pred = ensemble_decision

    elif new_model is not None: #two-way
        _, new_pred,_  = new_model.eval(dataloader)

    new_correct_mask = (gt == new_pred)
    new_incorrect_mask = ~new_correct_mask
    new_correct_indices = indices[new_correct_mask]
    new_incorrect_indices = indices[new_incorrect_mask]

    base_error = len(np.intersect1d(new_incorrect_indices, old_incorrect_indices))
    undesired_error = len(np.intersect1d(new_incorrect_indices, old_correct_indices))
    desired_non_error = len(np.intersect1d(new_correct_indices, old_incorrect_indices))
    base_non_error = len(np.intersect1d(new_correct_indices, old_correct_indices))
    logger.info('{}, {}'.format(base_error, desired_non_error))
    logger.info('{}, {}'.format(undesired_error, base_non_error))
    
    logger.info('ACC compare: {}, {}'.format(new_correct_mask.mean()*100, old_correct_mask.mean()*100))

# def build_info(dataset_splits: Dataset.DataSplits, name, clf:Detector.Prototype, old_batch_size, new_batch_size):
#     data_info = {}
#     assert name != 'train', 'Avoid dataloader shuffles!'
#     weakness_score, _ = clf.predict(dataset_splits.loader[name])        
#     data_info['weakness_score'] = weakness_score
#     data_info['old_batch_size'] = old_batch_size
#     data_info['new_batch_size'] = new_batch_size
#     data_info['dataset'] = dataset_splits.dataset[name]
#     data_info['loader'] = dataset_splits.loader[name]
#     logger.info('Set Up Test Loader Info')
#     return data_info

# def update_info(data_info, clf:Detector.Prototype):
#     weakness_score, _ = clf.predict(data_info['loader'])        
#     data_info['weakness_score'] = weakness_score
#     return data_info
    
def get_correctness_weakness_score(model: Model.Prototype, dataloader, clf:Detector.Prototype, correctness):
    '''
    weakness score of data w.r.t. the correctness of a given model
    '''
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    weakness_score, _ = clf.predict(dataloader)
    if correctness:
        mask = (dataset_gts == dataset_preds)
    else:
        mask = (dataset_gts != dataset_preds)
    return weakness_score[mask]

