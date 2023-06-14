from utils import config
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.log as Log
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
from abc import abstractmethod
import numpy as np
import torch

class subset_setter():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_subset_loders(self, data_info):
        '''
        data_info: dict of data, gt and pred labels, batch_size, and dv(if needed)
        '''
        pass

class disrtibution():
    def __init__(self, prior, dstr) -> None:
        self.prior = prior
        self.dstr = dstr

class probability_setter(subset_setter):
    def __init__(self) -> None:
        super().__init__()

    def go_to_dstr(self, value, target_dstr: disrtibution, other_dstr:disrtibution):
        probability_1 = (target_dstr.prior * target_dstr.dstr.pdf(value)) / (target_dstr.prior * target_dstr.dstr.pdf(value) + other_dstr.prior * other_dstr.dstr.pdf(value))
        if probability_1 >= 0.5:
            return True
        else:
            return False

    def get_subset_loders(self, data_info, correct_dstr: disrtibution, incorrect_dstr: disrtibution):
        selected_mask = []
        for value in data_info['dv']:
            selected_decision = self.go_to_dstr(value, incorrect_dstr, correct_dstr)
            selected_mask.append(selected_decision)
        selected_mask = np.array(selected_mask)
        dataset_indices = np.arange(len(data_info['dataset']))
        test_selected = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['new_batch_size'], num_workers=config['num_workers'])
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=config['num_workers'])       
        test_loader = {
            'new_model':test_selected_loader,
            'old_model': remained_test_loader
        }   
        print('selected test images: {}%'.format(np.round(len(test_selected)/len(data_info['dv']), decimals=3)*100))
        print('new cls percent:', new_cls_stat(test_selected))
        print('the max dv:', np.max(data_info['dv'][dataset_indices[selected_mask]]))
        return test_loader
    
class threshold_subset_setter(subset_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self, data_info, threshold):
        print(threshold)
        selected_mask = data_info['dv'] <= threshold
        dataset_indices = np.arange(len(data_info['dataset']))
        test_selected = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['new_batch_size'], num_workers=config['num_workers'])
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=config['num_workers'])
        test_loader = {
            'new_model':test_selected_loader,
            'old_model': remained_test_loader
        }   
        print('selected test images:', len(test_selected))
        return test_loader
class misclassification_subset_setter(subset_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self,data_info):
        incorr_cls_indices = (data_info['gt'] != data_info['pred'])
        corr_cls_indices = (data_info['gt'] == data_info['pred'])
        incorr_cls_set = torch.utils.data.Subset( data_info['dataset'],np.arange(len(data_info['dataset']))[incorr_cls_indices])
        corr_cls_set = torch.utils.data.Subset( data_info['dataset'],np.arange(len(data_info['dataset']))[corr_cls_indices])
        corr_cls_loader = torch.utils.data.DataLoader(corr_cls_set, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        incorr_cls_loader = torch.utils.data.DataLoader(incorr_cls_set, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        test_loader = {
            'new_model':incorr_cls_loader,
            'old_model': corr_cls_loader
        }   
        subset_loader = [test_loader]
        return subset_loader
    
def get_threshold(clf:Detector.SVM , acquisition_config:Config.Acquistion, model_config:Config.NewModel, data_splits:Dataset.DataSplits):
    if 'seq' in acquisition_config.method: 
        clf = Log.get_log_clf(acquisition_config, model_config, data_splits.loader['train_clip'], clf.clip_processor)
    new_data = Log.get_log_data(acquisition_config, model_config, data_splits)
    train_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                            num_workers= config['num_workers'])
    train_dv, _ = clf.predict(train_data_loader)
    return np.max(train_dv)

def mis_cls_stat(split_name, data_split:Dataset.DataSplits, model:Model.prototype):
    '''
    Get misclassification proportion on some labels
    '''
    check_labels = config['data']['remove_fine_labels']
    gt,pred,_  = model.eval(data_split.loader[split_name])
    dataset = data_split.dataset[split_name]
    dataset_idx = np.arange(len(dataset))
    incor_mask = (gt!=pred)
    incor_idx = dataset_idx[incor_mask]
    incor_dataset = torch.utils.data.Subset(dataset,incor_idx)
    incor_fine_labels = Dataset.get_ds_labels(incor_dataset, use_fine_label=True)
    check_labels_cnt = 0
    for label in check_labels:
        check_labels_cnt += (label==incor_fine_labels).sum()
    check_labels_cnt = check_labels_cnt/len(incor_fine_labels) * 100
    return check_labels_cnt

def new_cls_stat(dataset):
    check_labels = config['data']['remove_fine_labels']
    ds_labels = Dataset.get_ds_labels(dataset, use_fine_label=True)
    check_labels_cnt = 0
    for label in check_labels:
        check_labels_cnt += (label==ds_labels).sum()
    check_labels_cnt = check_labels_cnt/len(ds_labels) * 100
    return check_labels_cnt
