from utils import config
import utils.objects.Config as Config
from abc import abstractmethod
import numpy as np
import torch
from utils.statistics.distribution import disrtibution

class Prototype():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def run(self, data_info):
        '''
        data_info: dict of data, gt and pred labels, batch_size, and dv(if needed)
        '''
        pass

class Probability(Prototype):
    def __init__(self) -> None:
        super().__init__()

    def get_posterior(self, value, dstr_dict:dict, pdf_type):
        '''
        Get Posterior of Target Dstr in dstr_dict
        '''
        target_dstr = dstr_dict['target']
        other_dstr = dstr_dict['other']
        if pdf_type == 'norm':
            target_posterior = self.norm(value, target_dstr, other_dstr)  
        else:
            target_posterior = self.kde(value, target_dstr, other_dstr)
        return target_posterior
    
    def norm(self, value, target_dstr: disrtibution, other_dstr:disrtibution):
        probability_target = (target_dstr.prior * target_dstr.dstr.pdf(value)) / (target_dstr.prior * target_dstr.dstr.pdf(value) + other_dstr.prior * other_dstr.dstr.pdf(value))
        return probability_target
   
    def kde(self, value, target_dstr: disrtibution, other_dstr:disrtibution):
        probability_target = (target_dstr.prior * target_dstr.dstr.evaluate(value)) / (target_dstr.prior * target_dstr.dstr.evaluate(value) + other_dstr.prior * other_dstr.dstr.evaluate(value))
        # probability_target = (target_dstr.prior * target_dstr.dstr.pdf(value)) / (target_dstr.prior * target_dstr.dstr.pdf(value) + other_dstr.prior * other_dstr.dstr.pdf(value))

        return probability_target

    def run(self, data_info, dstr_dict:dict, stream_instruction:Config.ProbabStream):
        selected_probab = []
        for value in data_info['dv']:
            target_posterior = self.get_posterior(value, dstr_dict, stream_instruction.pdf)
            selected_probab.append(target_posterior)
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
    
class Threshold(Prototype):
    def __init__(self) -> None:
        pass
    def run(self, data_info, threshold):
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
class Mistakes(Prototype):
    def __init__(self) -> None:
        pass
    def run(self,data_info):
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
    