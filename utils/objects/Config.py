import os
from abc import abstractmethod

class Ensemble():
    def __init__(self, criterion=None, name=None, pdf=None) -> None:
        self.name = name
        self.criterion = criterion
        self.pdf =pdf # probability density function

# class WTAEnsemble(Ensemble):
#     def __init__(self, criterion, name, pdf) -> None:
#         super().__init__(criterion, name)
#         self.pdf = pdf

# class DVEnsemble(Ensemble):
#     def __init__(self, criterion, name) -> None:
#         super().__init__(criterion, name)

class Detection():
    def __init__(self, name, vit_mounted) -> None:
        self.name = name
        self.vit = vit_mounted

class Acquisition():
    def __init__(self, method='', budget=0, threshold=-1, seq_config=None) -> None:
        self.method = method
        self.budget = budget
        self.threshold = None if threshold == -1 else threshold
        self.seq_config = seq_config

    @abstractmethod
    def get_new_data_size(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

    def set_budget(self, budget):
        self.budget = budget

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_method(self, method):
        self.method = method
    
class NonSeqAcquisition(Acquisition):
    def __init__(self, method='', budget=0, threshold=None, seq_config=None) -> None:
        super().__init__(method, budget, threshold, seq_config)

    def get_new_data_size(self, class_number):
        return class_number*self.budget
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{}'.format(self.method, self.budget)

class SequentialAc(Acquisition):
    def __init__(self, method='', budget=0, threshold=None, seq_config=None) -> None:
        super().__init__(method, budget, threshold, seq_config)

    def set_up(self):
        self.round_acquire_method = 'pd' if 'pd' in self.method else 'dv'
        if self.seq_config['budget'] != None:
            self.budget_round = self.seq_config['budget']
            self.n_rounds = self.budget // self.budget_round
            self.seq_mode = 'by_budget'
        else:
            self.n_rounds = self.seq_config['n_rounds']
            self.current_round = 0
            self.budget_non_last_round = self.budget  // self.n_rounds
            consumed_budget = self.budget_non_last_round * (self.n_rounds - 1)
            self.budget_last_round = self.budget - consumed_budget
            self.seq_mode = 'by_round'
        
    def get_new_data_size(self, class_number):
        return class_number * self.budget_last_round + class_number * self.budget_non_last_round * (self.n_rounds-1)
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:({},{}) in round {}'.format(
            self.method, self.budget_non_last_round, self.budget_last_round, self.current_round)

def AcquisitionFactory(acquisition_method, data_config):
    if 'seq' in acquisition_method:
        return SequentialAc(method=acquisition_method, seq_config=data_config['seq'])
    else:
        return NonSeqAcquisition(method=acquisition_method)
    
class Operation():
    '''
    Acquire + Ensemble + Detection 
    '''
    def __init__(self, acquisition: Acquisition, ensemble: Ensemble, detection: Detection) -> None:
        self.acquisition = acquisition
        self.ensemble = ensemble
        self.detection = detection

def check_dir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

class Model():
    def __init__(self, batch_size, class_number, model_dir, device, base_type) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.model_dir = model_dir
        self.root = os.path.join('model', model_dir, base_type, str(batch_size))
        check_dir(self.root)
        self.device = device
        self.base_type = base_type
        self.path = None
            
class OldModel(Model):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, base_type) -> None:
        super().__init__(batch_size, class_number, model_dir, device, base_type)
        self.model_cnt = model_cnt # can be redundant if dv_stat_test not epoch-wised
        self.path = os.path.join(self.root,'{}.pt'.format(model_cnt))

class NewModel(Model):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, pure:bool, setter, new_batch_size, base_type) -> None:
        super().__init__(batch_size, class_number, model_dir, device, base_type)
        self.pure = pure
        self.setter = setter
        self.new_batch_size = new_batch_size
        self.set_root(model_cnt)
        # root_detector = None

    def detector2root(self, acquisition_method, detector_name):
        # Make Conf and sampling-based method Root agnostic to detector
        if acquisition_method in ['conf', 'rs']:
            temp_root = os.path.join(self.root, 'no-detector')
        else:
            temp_root = os.path.join(self.root, detector_name)
        check_dir(temp_root)
        return temp_root
    
    def set_path(self, operation:Operation):
        # # Set Seq Acquisition Root
        # if 'seq' in acquisition_config.method:
        #     root = self.set_seq_root(self.root, acquisition_config)
        # else:
        #     root = self.root
        self.root_detector = self.detector2root(operation.acquisition.method, operation.detection.name)
        threshold_name = '_{}%'.format(int(operation.acquisition.threshold*100)) if operation.acquisition.threshold != None else ''
        self.path = os.path.join(self.root_detector, '{}_{}{}.pt'.format(operation.acquisition.method, operation.acquisition.budget, threshold_name))
    
    def set_root(self, model_cnt):
        pure_name = 'pure' if self.pure else 'non-pure'
        if self.base_type != 'svm':
            self.root = os.path.join(self.root, self.setter, pure_name, str(model_cnt), str(self.new_batch_size)) 
        else:
            self.root = os.path.join(self.root, pure_name, str(model_cnt)) 
        check_dir(self.root)

    def set_seq_root(self,root, acquisition_config:SequentialAc):
        # root = os.path.join(root,'{}_rounds'.format(acquisition_config.sequential_rounds_info[acquisition_config.budget]))
        check_dir(root)
        return root

def str2bool(value):
    if isinstance(value,bool):
        return value
    else:
        return False if value=='0' else True
    
def str2float(value):
    return float(value)