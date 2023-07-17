import os
from utils import config
from abc import abstractmethod
import torch
# import utils.objects.Detector as Detector

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

class Dectector():
    def __init__(self, name, vit_mounted) -> None:
        self.name = name
        self.vit = vit_mounted

class Acquistion():
    method: str
    n_ndata:int
    bound: float

    def __init__(self) -> None:
        self.method = ''
        self.n_ndata = 0
        self.bound = None

    @abstractmethod
    def get_new_data_size(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

    def add_streaming(self, streamer:Stream):
        self.stream = streamer

    def add_detector(self, detector:Dectector):
        self.detector = detector

class NonSeqAcquistion(Acquistion):
    def __init__(self) -> None:
        super().__init__()

    def get_new_data_size(self, class_number):
        return class_number*self.n_ndata
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{}'.format(self.method, self.n_ndata)

class SequentialAc(Acquistion):
    def __init__(self, sequential_rounds:dict) -> None:
        super().__init__()
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
            self.method, self.n_data_not_last, self.n_data_last_round, self.current_round)

def AcquistionFactory(strategy, sequential_rounds_config):
    if strategy == 'non_seq':
        return NonSeqAcquistion()
    else:
        return SequentialAc(sequential_rounds_config)
    
def check_dir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

class ModelConfig():
    def __init__(self, batch_size, class_number, model_dir, device, base_type) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.model_dir = model_dir
        self.root = os.path.join(config['base_root'], model_dir, base_type, str(batch_size))
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
        self.root_detector = None

    def detector2root(self, acquistion_config:Acquistion):
        # Make Conf and sampling-based method Root agnostic to detector
        if acquistion_config.method in ['conf', 'sm']:
            temp_root = os.path.join(self.root, 'no-detector')
        else:
            temp_root = os.path.join(self.root, acquistion_config.detector.name)
        check_dir(temp_root)
        return temp_root
    
    def set_path(self, acquistion_config:Acquistion):
        # # Set Seq Acquistion Root
        # if 'seq' in acquistion_config.method:
        #     root = self.set_seq_root(self.root, acquistion_config)
        # else:
        #     root = self.root
        self.root_detector = self.detector2root(acquistion_config)
        bound_name = '_{}'.format(acquistion_config.bound) if acquistion_config.bound != None else ''
        self.path = os.path.join(self.root_detector, '{}_{}{}.pt'.format(acquistion_config.method, acquistion_config.n_ndata, bound_name))
    
    def set_root(self, model_cnt):
        pure_name = 'pure' if self.pure else 'non-pure'
        if self.base_type != 'svm':
            self.root = os.path.join(self.root, self.setter, pure_name, str(model_cnt), str(self.new_batch_size)) 
        else:
            self.root = os.path.join(self.root, pure_name, str(model_cnt)) 
        check_dir(self.root)

    def set_seq_root(self,root, acquistion_config:SequentialAc):
        # root = os.path.join(root,'{}_rounds'.format(acquistion_config.sequential_rounds_info[acquistion_config.n_ndata]))
        check_dir(root)
        return root

def str2bool(value):
    if isinstance(value,bool):
        return value
    else:
        return False if value=='0' else True
    
def str2float(value):
    return float(value)

def parse():
    hparams = config['hparams']
    base_batch_size = hparams['batch_size']['base']
    new_batch_size = hparams['batch_size']['new']
    data_config = config['data']
    label_map = data_config['label_map']
    n_new_data = data_config['n_new_data']
    img_per_cls_list = n_new_data
    superclass_num = hparams['superclass']
    ratio = data_config['ratio']
    seq_rounds = 2
    train_labels = data_config['train_label']
    batch_size = {
        'base': base_batch_size,
        'new': new_batch_size
    }
    output = {
        'label_map': label_map,
        'ratio': ratio,
        'removed_labels': config['data']['remove_fine_labels'],
        'svm_kernel': config['clf_args']['kernel'],
        'superclass': superclass_num,
    }
    print(output)
    return batch_size, train_labels, label_map, img_per_cls_list, superclass_num, ratio, seq_rounds 