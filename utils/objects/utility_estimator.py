import utils.objects.Detector as Detector
import numpy as np
import utils.statistics.distribution as distribution_utils
from utils.logging import logger
from typing import Dict
from abc import abstractmethod
from utils.statistics.distribution import CorrectnessDisrtibution
import utils.objects.model as Model
import utils.objects.Config as Config
class Base():
    '''
    Convert weakness feature scores from the detector to data utility
    '''
    def __init__(self) -> None:
        pass

    @ abstractmethod
    def set_up(self):
        pass

    @abstractmethod
    def run(self, dataloader):
        pass

class U_WFS(Base):
    def __init__(self) -> None:
        super().__init__()

    def set_up(self, detector: Detector.Prototype):
        self.detector = detector
    
    def run(self, dataloader):
        feature_scores, _ = self.detector.predict(dataloader)
        return feature_scores

class U_WFSD(Base):
    def __init__(self) -> None:
        super().__init__()

    def set_up(self, detector: Detector.Prototype, validation_loader, pdf, base_model):
        self.detector = detector
        correct_dstr = distribution_utils.CorrectnessDisrtibution(base_model, self.detector, validation_loader, pdf, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(base_model, self.detector, validation_loader, pdf, correctness=False)
        self.wfs_dstr = {'correct': correct_dstr, 'incorrect': incorrect_dstr}

    def get_posterior(self, value, dstr_dict: Dict[str, CorrectnessDisrtibution]):
        '''
        Get Posterior of Target Dstr in dstr_dict
        '''
        target_dstr = dstr_dict['target']
        other_dstr = dstr_dict['other']
        return (target_dstr.prior * target_dstr.pdf.evaluate(value)) / (target_dstr.prior * target_dstr.pdf.evaluate(value) + other_dstr.prior * other_dstr.pdf.evaluate(value))
    
    def get_posterior_list(self, dstr_dict: Dict[str, distribution_utils.CorrectnessDisrtibution], observations):
        probabilities = []
        for value in observations:
            posterior = self.get_posterior(value, dstr_dict)
            probabilities.append(posterior)
        return np.concatenate(probabilities)
   
    def run(self, dataloader):
        feature_scores, _ = self.detector.predict(dataloader)
        utility_score = self.get_posterior_list({'target': self.wfs_dstr['incorrect'], 'other': self.wfs_dstr['correct']}, feature_scores)
        return utility_score

class Confidence(Base):
    def __init__(self) -> None:
        super().__init__()

    def set_up(self):
        return super().set_up()
    
    def get_gt_probab(self, gts, probab):
        '''
        Return Prediction Probability of True Labels \n
        probab: (n_samples, n_class)
        '''
        return probab[np.arange(len(gts)), gts]

    def get_gt_distance(self, gts, decision_values):
        '''
        Return Distance to HyperPlane of True Labels \n
        decision_values: (n_samples)
        '''
        cls_0_mask = (gts==0)
        cls_1_mask = ~cls_0_mask
        distance = np.zeros(len(gts))
        distance[cls_0_mask] = (0 - decision_values[cls_0_mask])
        distance[cls_1_mask] = (decision_values[cls_1_mask])
        return distance[gts]
    
    def run(self,  base_model_config: Config.OldModel, base_model:Model.Prototype, market_loader):

        market_gts, _, market_score = base_model.eval(market_loader)

        if base_model_config.base_type == 'svm':
            confs = self.get_gt_distance(market_gts, market_score)
        else:
            confs = self.get_gt_probab(market_gts, market_score)
        return confs 
    
class Entropy(Base):
    def __init__(self) -> None:
        super().__init__()

    def set_up(self):
        return super().set_up()
    
    def get_entropy(self, probabs):
        entropy = -np.sum(probabs * np.log2(probabs), axis=1)
        return entropy

    def run(self, base_model:Model.Prototype, dataloader):

        _, _, probabs = base_model.eval(dataloader)

        return self.get_entropy(probabs)
      
def factory(estimator_name, detector: Detector.Prototype, validation_loader, pdf, base_model):
    if estimator_name == 'u-ws':
        estimator = U_WFS()
        estimator.set_up(detector)
        logger.info('U-WS')
    elif estimator_name == 'u-wsd':
        estimator = U_WFSD()
        estimator.set_up(detector, validation_loader, pdf, base_model)
        logger.info('U-WSD')
    else:
        logger.info('Utility Estimator is not Implemented.')
        exit()

    return estimator
    