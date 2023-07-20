from utils import config
import utils.objects.model as Model
import utils.objects.Config as Config
from utils.objects.log import Log
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
import numpy as np
import torch

def get_new_data_max_dv(clf:Detector.SVM , acquisition_config:Config.Acquisition, model_config:Config.NewModel, data_splits:Dataset.DataSplits):
    if 'seq' in acquisition_config.method: 
        log = Log(model_config, 'clf')
        clf = log.import_log(acquisition_config)    
    log = Log(model_config, 'indices')
    new_data_indices = log.import_log(acquisition_config)
    new_data = torch.utils.data.Subset(data_splits.dataset['market'], new_data_indices)    
    train_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                            num_workers= config['num_workers'])
    train_dv, _ = clf.predict(train_data_loader)
    return np.max(train_dv)

def label_stat(dataset, checked_labels):
    ds_labels = Dataset.get_ds_labels(dataset, use_fine_label=True)
    check_labels_cnt = 0
    for label in checked_labels:
        check_labels_cnt += (label==ds_labels).sum()
    # check_labels_cnt = check_labels_cnt/len(ds_labels) * 100
    # return check_labels_cnt
    return check_labels_cnt

def mis_label_stat(split_name, data_split:Dataset.DataSplits, model:Model.prototype):
    '''
    Get misclassification proportion on target labels
    '''
    check_labels = config['data']['remove_fine_labels']
    gt,pred,_  = model.eval(data_split.loader[split_name])
    dataset = data_split.dataset[split_name]
    dataset_idx = np.arange(len(dataset))
    incor_mask = (gt!=pred)
    incor_idx = dataset_idx[incor_mask]
    incor_dataset = torch.utils.data.Subset(dataset,incor_idx)
    return label_stat(incor_dataset, check_labels) / len(incor_dataset) * 100

def pred_metric(dataloader, old_model:Model.prototype, new_model:Model.prototype):
    gt,pred,_  = old_model.eval(dataloader)
    indices = np.arange(len(gt))
    old_correct_mask = (gt == pred)
    old_incorrect_mask = ~old_correct_mask
    old_correct_indices = indices[old_correct_mask]
    old_incorrect_indices = indices[old_incorrect_mask]

    gt,pred,_  = new_model.eval(dataloader)
    new_correct_mask = (gt == pred)
    new_incorrect_mask = ~new_correct_mask
    new_correct_indices = indices[new_correct_mask]
    new_incorrect_indices = indices[new_incorrect_mask]

    # print(old_correct_mask.sum(), old_incorrect_mask.sum())
    # print(new_correct_mask.sum(), new_incorrect_mask.sum())

    tn = len(np.intersect1d(new_incorrect_indices, old_incorrect_indices))
    fn = len(np.intersect1d(new_incorrect_indices, old_correct_indices))
    tp = len(np.intersect1d(new_correct_indices, old_incorrect_indices))
    fp = len(np.intersect1d(new_correct_indices, old_correct_indices))
    print(tn, tp)
    print(fn, fp)

    # dataset = loader2dataset(dataloader)
    # old_incor_data = torch.utils.data.Subset(dataset, old_incorrect_indices)
    # new_cor_data = torch.utils.data.Subset(dataset, new_correct_indices)
    # new_cor_old_incor_data = torch.utils.data.Subset(dataset, np.intersect1d(new_correct_indices, old_incorrect_indices))
    # print(label_stat(old_incor_data, config['data']['remove_fine_labels']), label_stat(new_cor_old_incor_data, config['data']['remove_fine_labels']), label_stat(new_cor_data, config['data']['remove_fine_labels']))

    # old_labels = set(config['data']['train_label']) - set(config['data']['remove_fine_labels'])
    # print(label_stat(old_incor_data, old_labels), label_stat(new_cor_old_incor_data, old_labels), label_stat(new_cor_data, old_labels))


def build_info(dataset_splits: Dataset.DataSplits, name, clf:Detector.Prototype, old_batch_size, new_batch_size, base_model:Model.prototype):
    data_info = {}
    dv, _ = clf.predict(dataset_splits.loader[name], base_model)        
    data_info['dv'] = dv
    data_info['old_batch_size'] = old_batch_size
    data_info['new_batch_size'] = new_batch_size
    data_info['dataset'] = dataset_splits.dataset[name]
    return data_info
    
def get_hard_easy_dv(model: Model.prototype, dataloader, clf:Detector.Prototype):
    '''
    DV of hard and easy data wrt the given model
    '''
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    dv, _ = clf.predict(dataloader, model)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    cor_dv = dv[cor_mask]
    incor_dv = dv[incor_mask]
    return cor_dv, incor_dv
