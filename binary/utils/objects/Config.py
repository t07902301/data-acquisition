import os
from abc import abstractmethod

class Stream():
    def __init__(self, bound, name) -> None:
        self.name = name
        self.bound = bound

class ProbabStream(Stream):
    def __init__(self, bound, name, pdf) -> None:
        super().__init__(bound, name)
        self.pdf = pdf

class DVStream(Stream):
    def __init__(self, bound, name) -> None:
        super().__init__(bound, name)

class Detection():
    def __init__(self, name, vit_mounted) -> None:
        self.name = name
        self.vit = vit_mounted

class Acquisition():
    def __init__(self, method='', n_ndata=0, bound=None) -> None:
        self.method = method
        self.n_ndata = n_ndata
        self.bound = bound

    @abstractmethod
    def get_new_data_size(self):
        pass

    @abstractmethod
    def get_info(self):
        pass
    
class NonSeqAcquisition(Acquisition):
    def __init__(self, method='', n_ndata=0, bound=None) -> None:
        super().__init__(method, n_ndata, bound)

    def get_new_data_size(self, class_number):
        return class_number*self.n_ndata
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{}'.format(self.method, self.n_ndata)

class SequentialAc(Acquisition):
    def __init__(self, method='', n_ndata=0, bound=None, sequential_rounds:dict=None) -> None:
        super().__init__(method, n_ndata, bound)
        # self.n_rounds_info = sequential_rounds

    def set_up(self):
        self.n_rounds = 3
        self.round_acquire_method = 'dv'
        self.current_round = 0
        self.n_data_non_last_round = self.n_ndata  // self.n_rounds
        n_data_acquired = self.n_data_non_last_round * (self.n_rounds - 1)
        self.n_data_last_round = self.n_ndata - n_data_acquired
        self.n_data_round = 0
        
    def set_round(self, round):
        self.current_round = round + 1
        if self.current_round == self.n_rounds:
            self.n_data_round = self.n_data_last_round
        else:
            self.n_data_round = self.n_data_non_last_round

    def get_new_data_size(self, class_number):
        return class_number * self.n_data_last_round + class_number * self.n_data_non_last_round * (self.n_rounds-1)
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:({},{}) in round {}'.format(
            self.method, self.n_data_non_last_round, self.n_data_last_round, self.current_round)

def AcquisitionFactory(method, acquisition:Acquisition):
    if 'seq' in method:
        return SequentialAc(acquisition.method, acquisition.n_ndata, acquisition.bound)
    else:
        return NonSeqAcquisition(acquisition.method, acquisition.n_ndata, acquisition.bound)
    
class Operation():
    '''
    Acquire + Stream + Detection 
    '''
    def __init__(self, acquisition: Acquisition, stream: Stream, detection: Detection) -> None:
        self.acquisition = acquisition
        self.stream = stream
        self.detection = detection

def check_dir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

class ModelConfig():
    def __init__(self, batch_size, class_number, model_dir, device, base_type) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.model_dir = model_dir
        self.root = os.path.join('model/', model_dir, base_type, str(batch_size))
        check_dir(self.root)
        self.device = device
        self.base_type = base_type
        self.path = None
            
class OldModel(ModelConfig):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, base_type) -> None:
        super().__init__(batch_size, class_number, model_dir, device, base_type)
        self.model_cnt = model_cnt # can be redundant if dv_stat_test not epoch-wised
        self.path = os.path.join(self.root,'{}.pt'.format(model_cnt))

class NewModel(ModelConfig):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, pure:bool, setter, new_batch_size, base_type) -> None:
        super().__init__(batch_size, class_number, model_dir, device, base_type)
        self.pure = pure
        self.setter = setter
        self.new_batch_size = new_batch_size
        self.set_root(model_cnt)
        # root_detector = None

    def detector2root(self, acquisition_method, detector_name, stream_bound=None):
        # Make Conf and sampling-based method Root agnostic to detector
        if acquisition_method in ['conf', 'sm']:
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
        self.root_detector = self.detector2root(operation.acquisition.method, operation.detection.name, operation.stream.bound)
        bound_name = '_{}'.format(operation.acquisition.bound) if operation.acquisition.bound != None else ''
        self.path = os.path.join(self.root_detector, '{}_{}{}.pt'.format(operation.acquisition.method, operation.acquisition.n_ndata, bound_name))
    
    def set_root(self, model_cnt):
        pure_name = 'pure' if self.pure else 'non-pure'
        if self.base_type != 'svm':
            self.root = os.path.join(self.root, self.setter, pure_name, str(model_cnt), str(self.new_batch_size)) 
        else:
            self.root = os.path.join(self.root, pure_name, str(model_cnt)) 
        check_dir(self.root)

    def set_seq_root(self,root, acquisition_config:SequentialAc):
        # root = os.path.join(root,'{}_rounds'.format(acquisition_config.sequential_rounds_info[acquisition_config.n_ndata]))
        check_dir(root)
        return root
    
    def check_rs(self, acquisition_method, stream_bound):
        if acquisition_method == 'sm' and stream_bound == 0:
            return True
        else:
            return False

def str2bool(value):
    if isinstance(value,bool):
        return value
    else:
        return False if value=='0' else True
    
def str2float(value):
    return float(value)