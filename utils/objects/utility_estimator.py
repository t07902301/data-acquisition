import utils.objects.Detector as Detector
import numpy as np
import utils.statistics.distribution as distribution_utils
from utils.logging import logger
from typing import Dict
from abc import abstractmethod
from utils.statistics.distribution import CorrectnessDisrtibution

class Base():
    '''
    Convert weakness feature scores from the detector to data utility
    '''
    def __init__(self) -> None:
        self.detector = None

    @ abstractmethod
    def set_up(self, detector: Detector.Prototype):
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
        self.anchor_dstr = {'correct': correct_dstr, 'incorrect': incorrect_dstr}

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
        utility_score = self.get_posterior_list({'target': self.anchor_dstr['incorrect'], 'other': self.anchor_dstr['correct']}, feature_scores)
        return utility_score

def factory(estimator_name, detector: Detector.Prototype, validation_loader, pdf, base_model):
    if estimator_name == 'u-wfs':
        estimator = U_WFS()
        estimator.set_up(detector)
        logger.info('U-WFS')
    elif estimator_name == 'u-wfsd':
        estimator = U_WFSD()
        estimator.set_up(detector, validation_loader, pdf, base_model)
        logger.info('U-WFSD')
    else:
        logger.info('Unimplemented Utility Estimator.')
        exit()

    return estimator
    