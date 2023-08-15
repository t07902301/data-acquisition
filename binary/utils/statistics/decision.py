import utils.objects.model as Model
import numpy as np
from abc import abstractmethod

class Prototye():
    def __init__(self,class_number=1) -> None:
        self.class_numer = class_number
    @abstractmethod
    def get(self, model: Model.prototype, dataloader):
        '''
        Get a Model's decision
        '''
        pass
    @abstractmethod
    def apply(self, ensembled_decision):
        '''
        Get final decision from ensembled results
        '''
        pass
    
class Confidence(Prototye):
    def __init__(self, class_number=1) -> None:
        super().__init__(class_number)

    def get(self, model: Model.CNN, dataloader):
        _, _, probab  = model.eval(dataloader)
        if self.class_numer == 1:
            probab = self.transform_BDE_probab(probab)
        return probab
    
    def transform_BDE_probab(self, class_1_probab):
        '''
            Make 1D sigmoid output be 2D
        '''
        size = len(class_1_probab)
        class_0_probab = (1-class_1_probab).reshape((size,1))
        class_1_probab = class_1_probab.reshape((size,1))
        return np.concatenate((class_0_probab, class_1_probab), axis=1)
    
    def apply(self, ensembled_decision):
        # ensembled_decision: array (n_size, n_class) - multi-class
        preds = np.argmax(ensembled_decision, axis=-1)
        return preds
    
class Distance(Prototye):
    def __init__(self, class_number=1) -> None:
        super().__init__(class_number)

    def get(self, model: Model.svm, dataloader):
        _, _, distance = model.eval(dataloader)
        size = len(distance)
        distance = np.array(distance).reshape((size, 1))
        return distance
    
    def apply(self, ensembled_decision):
        # ensembled_decision: array (n_size, n_class) - binary
        size = len(ensembled_decision)
        ensembled_decision = ensembled_decision.reshape(size)
        class_1_mask = (ensembled_decision >= 0)
        preds = np.zeros(size)
        preds[class_1_mask] = 1
        return preds
    
def factory(model_type,class_number):
    if model_type == 'svm':
        return Distance(class_number)
    else:
        return Confidence(class_number)