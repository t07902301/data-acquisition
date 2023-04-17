from utils import n_workers
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.CLF as CLF
import utils.log as log
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
class threshold_subset_setter(subset_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self, data_info, threshold):
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices = acquistion.extract_class_indices(c, data_info['gt'])
            cls_dv = data_info['dv'][cls_indices]
            dv_selected_indices = (cls_dv<=threshold[c])
            selected_indices_total.append(dv_selected_indices)
        selected_indices_total = np.concatenate(selected_indices_total)
        test_selected = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[selected_indices_total])
        remained_test = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[~selected_indices_total])
        test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['batch_size'], num_workers=n_workers)
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['batch_size'], num_workers=n_workers)
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
        corr_cls_loader = torch.utils.data.DataLoader(corr_cls_set, batch_size=data_info['batch_size'], num_workers=n_workers)
        incorr_cls_loader = torch.utils.data.DataLoader(incorr_cls_set, batch_size=data_info['batch_size'], num_workers=n_workers)
        test_loader = {
            'new_model':incorr_cls_loader,
            'old_model': corr_cls_loader
        }   
        subset_loader = [test_loader]
        return subset_loader
    
def get_threshold(clf, clip_processor, acquisition_config:Config.Acquistion, model_config:Config.NewModel, market_loader):
    '''
    Use indices log and SVM to determine max decision values for each class.\n
    old_model + data -> SVM \n
    model_config + acquisition -> indices log
    '''
    if acquisition_config.method == 'seq_clf':
        return seq_bound(clf, clip_processor, acquisition_config, model_config)
    else:
        market_info = CLF.apply_CLF(clf, market_loader, clip_processor)
        return non_seq_bound(acquisition_config, model_config, market_info)

def non_seq_bound(acquisition_config:Config.Acquistion, model_config:Config.NewModel, market_info):
    idx_log_config = log.get_sub_log('indices', model_config, acquisition_config)
    idx_log_config.set_path(acquisition_config)
    new_data_indices = log.load(idx_log_config)
    max_dv = [np.max(market_info['dv'][new_data_indices[c]]) for c in range(model_config.class_number)]    
    return max_dv

def seq_bound(clf, clip_processor, acquisition_config:Config.Acquistion, model_config:Config.NewModel):
    data_log_config = log.get_sub_log('data', model_config, acquisition_config)
    data_log_config.set_path(acquisition_config)
    new_data = log.load(data_log_config)
    new_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                                    num_workers=n_workers)
    new_data_info = CLF.apply_CLF(clf, new_data_loader, clip_processor)
    max_dv = []
    for c in range(model_config.class_number):
        cls_indices = acquistion.extract_class_indices(c, new_data_info['gt'])
        cls_dv = new_data_info['dv'][cls_indices]
        max_dv.append(np.max(cls_dv))
    return max_dv

def get_max_dv(market_info,train_info,n_cls,new_data_indices,pure):
    '''
    Deprecated
    '''
    if pure:
        return [np.max(market_info['dv'][new_data_indices[c]]) for c in range(n_cls)]
    else:
        max_dv = []
        for c in range(n_cls):
            cls_indices = acquistion.extract_class_indices(c, train_info['gt'])
            train_cls_dv = train_info['dv'][cls_indices]
            cls_dv = np.concatenate((train_cls_dv, market_info['dv'][new_data_indices[c]]))
            max_dv.append(np.max(cls_dv))
        return max_dv

def seq_dv_bound(model_config, acquisition_config):
    clf_log = log.get_sub_log('clf', model_config, acquisition_config)
    data_log = log.get_sub_log('data', model_config, acquisition_config)
    clf_data = log.load(clf_log)
    data = log.load(data_log)
    model = Model.load(model_config)
    clf_data_loader ={
        split:torch.utils.data.DataLoader(ds, batch_size=model_config.batch_size, 
                                        num_workers=n_workers)  for split,ds in clf_data.items()
    }
    clf, clip, score = CLF.get_CLF(model, clf_data_loader)
    return score
    # data_loader = torch.utils.data.DataLoader(data, batch_size=model_config.batch_size, 
    #                                     num_workers=n_workers)
    # data_info = apply_CLF(clf, data_loader, clip)
    # max_dv = []
    # for c in range(model_config.class_number):
    #     cls_indices = acquistion.extract_class_indices(c, data_info['gt'])
    #     cls_dv = data_info['dv'][cls_indices]
    #     max_dv.append(np.max(cls_dv))
    # return max_dv
        

