import utils.objects.Config as Config
from abc import abstractmethod
import numpy as np
import torch
from utils.statistics.distribution import CorrectnessDisrtibution
from utils.dataset.wrappers import n_workers
from typing import Dict
from utils.logging import *
import utils.objects.Detector as Detector
import utils.objects.utility_estimator as ue

class Prototype():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def run(self, data_info):
        '''
        data_info: dict of data, gt and pred labels, batch_size, and weakness_score(if needed)
        '''
        pass

class Posterior(Prototype):
    def __init__(self) -> None:
        super().__init__()

    def get_posterior(self, value, dstr_dict: Dict[str, CorrectnessDisrtibution]):
        '''
        Get Posterior of Target Dstr in dstr_dict
        '''
        target_dstr = dstr_dict['target']
        other_dstr = dstr_dict['other']
        return (target_dstr.prior * target_dstr.pdf.evaluate(value)) / (target_dstr.prior * target_dstr.pdf.evaluate(value) + other_dstr.prior * other_dstr.pdf.evaluate(value))

    def run(self, data_info, dstr_dict: Dict[str, CorrectnessDisrtibution], ensemble_instruction:Config.Ensemble):
        '''
        Partition by posterior probab
        '''
        dataset_indices = np.arange(len(data_info['dataset']))
        posterior_list = []
        for idx in dataset_indices:
            target_posterior = self.get_posterior(data_info['weakness_score'][idx], dstr_dict)
            posterior_list.append(target_posterior)
        posterior_list = np.array(posterior_list).reshape((len(dataset_indices),))
        selected_mask = (posterior_list >= ensemble_instruction.criterion)
        selected_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        selected_test_loader = torch.utils.data.DataLoader(selected_test, batch_size=data_info['new_batch_size'], num_workers=n_workers)
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=n_workers)       
        test_loader = {
            'new_model':selected_test_loader,
            'old_model': remained_test_loader
        }   
        # logger.info('selected test images: {}%'.format(np.round(len(test_selected)/len(data_info['weakness_score']), decimals=3)*100))
        # logger.info('new cls percent:', new_label_stat(test_selected))
        # logger.info('the max weakness_score:', np.max(data_info['weakness_score'][dataset_indices[selected_mask]]))
        return test_loader, posterior_list

class ProbabWeaknessScore(Prototype):
    def __init__(self) -> None:
        super().__init__()
    def run(self, data_info, detector: Detector.Prototype, ensemble_instruction:Config.Ensemble):
        dataset_indices = np.arange(len(data_info['dataset']))
        probab, _ = detector.predict(torch.utils.data.DataLoader(data_info['dataset'], batch_size=data_info['new_batch_size'], num_workers=n_workers))
        probab = np.array(probab)
        selected_mask = (probab >= ensemble_instruction.criterion)
        selected_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        selected_test_loader = torch.utils.data.DataLoader(selected_test, batch_size=data_info['new_batch_size'], num_workers=n_workers)
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=n_workers)       
        test_loader = {
            'new_model':selected_test_loader,
            'old_model': remained_test_loader
        }   
        return test_loader, probab        
    
class DataValuation(Prototype):
    def __init__(self) -> None:
        pass
    def run(self, data_info, utility_estimator: ue.Base, criterion):
        dataset_indices = np.arange(len(data_info['dataset']))
        utility_score = utility_estimator.run(torch.utils.data.DataLoader(data_info['dataset'], batch_size=data_info['new_batch_size'], num_workers=n_workers))
        utility_score = np.array(utility_score)
        selected_mask = (utility_score >= criterion)
        dataset_indices = np.arange(len(data_info['dataset']))
        test_selected = torch.utils.data.Subset(data_info['dataset'],dataset_indices[selected_mask])
        remained_test = torch.utils.data.Subset(data_info['dataset'],dataset_indices[~selected_mask])
        selected_test_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['new_batch_size'], num_workers=n_workers)
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['old_batch_size'], num_workers=n_workers)
        test_loader = {
            'new_model':selected_test_loader,
            'old_model': remained_test_loader
        }   
        logger.info('selected test images:{}'.format(len(test_selected)))
        return test_loader
    