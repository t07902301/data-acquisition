from utils import config
import utils.objects.model as Model
import utils.objects.Config as Config
from utils.objects.log import Log
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
from abc import abstractmethod
import numpy as np
import torch
from utils.statistics.distribution import disrtibution

class subset_setter():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_subset_loders(self, data_info):
        '''
        data_info: dict of data, gt and pred labels, batch_size, and dv(if needed)
        '''
        pass

class probability_setter(subset_setter):
    def __init__(self) -> None:
        super().__init__()

    def get_probab(self, value, target_dstr: disrtibution, other_dstr:disrtibution, pdf_type):
        if pdf_type == 'norm':
            probab_target = self.norm_probab(value, target_dstr, other_dstr)  
        else:
            probab_target = self.kde_probab(value, target_dstr, other_dstr)
        return probab_target

    def norm_probab(self, value, target_dstr: disrtibution, other_dstr:disrtibution):
        probability_target = (target_dstr.prior * target_dstr.dstr.pdf(value)) / (target_dstr.prior * target_dstr.dstr.pdf(value) + other_dstr.prior * other_dstr.dstr.pdf(value))
        
        return probability_target
   
    def kde_probab(self, value, target_dstr: disrtibution, other_dstr:disrtibution):
        probability_target = (target_dstr.prior * target_dstr.dstr.evaluate(value)) / (target_dstr.prior * target_dstr.dstr.evaluate(value) + other_dstr.prior * other_dstr.dstr.evaluate(value))
        # probability_target = (target_dstr.prior * target_dstr.dstr.pdf(value)) / (target_dstr.prior * target_dstr.dstr.pdf(value) + other_dstr.prior * other_dstr.dstr.pdf(value))

        return probability_target

    def get_subset_loders(self, data_info, correct_dstr: disrtibution, incorrect_dstr: disrtibution, stream_instruction:Config.ProbabStream):
        selected_probab = []
        for value in data_info['dv']:
            probab_target = self.get_probab(value, incorrect_dstr, correct_dstr, stream_instruction.pdf)
            selected_probab.append(probab_target)
        selected_probab = np.array(selected_probab).reshape((len(selected_probab),))
        selected_mask = (selected_probab >= stream_instruction.bound)
        dataset_indices = np.arange(len(data_info['dataset']))
        test_selected = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['new_batch_size'], num_workers=config['num_workers'])
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=config['num_workers'])       
        test_loader = {
            'new_model':test_selected_loader,
            'old_model': remained_test_loader
        }   
        # print('selected test images: {}%'.format(np.round(len(test_selected)/len(data_info['dv']), decimals=3)*100))
        # print('new cls percent:', new_label_stat(test_selected))
        # print('the max dv:', np.max(data_info['dv'][dataset_indices[selected_mask]]))
        return test_loader, selected_probab
    
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
    
def get_threshold(clf:Detector.SVM , acquisition_config:Config.Acquisition, model_config:Config.NewModel, data_splits:Dataset.DataSplits):
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


def build_data_info(dataset_splits: Dataset.DataSplits, name, clf:Detector.Prototype, old_batch_size, new_batch_size, base_model:Model.prototype):
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
