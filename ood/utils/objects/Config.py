import os
from utils import config
from abc import abstractmethod
import re

class Acquistion():
    method: str
    n_ndata:int

    def __init__(self) -> None:
        self.method = ''
        self.n_ndata = 0

    def set_items(self, method, new_data_number):
        self.method = method
        self.n_ndata = new_data_number

    @abstractmethod
    def get_new_data_size(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

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
        self.sequential_rounds_info = sequential_rounds
        self.round_acquire_method = 'dv'
        self.current_round = 0
        self.n_data_last_round = 0
        self.n_data_not_last = 0
        self.round_data_per_class = 0
    def set_round(self, round):
        self.current_round = round + 1
        if self.current_round == self.sequential_rounds:
            self.round_data_per_class = self.n_data_last_round
        else:
            self.round_data_per_class = self.n_data_not_last
    def get_new_data_size(self, class_number):
        return class_number * self.n_data_last_round + class_number * self.n_data_not_last * (self.sequential_rounds-1)
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:({},{}) in round {}'.format(self.method, self.n_data_not_last, self.n_data_last_round, self.current_round)
    def set_items(self, method, new_data_number):
        super().set_items(method, new_data_number)
        # self.sequential_rounds = self.sequential_rounds_info[self.n_ndata]
        self.sequential_rounds = 2
        self.n_data_not_last = self.n_ndata  // self.sequential_rounds
        n_data_acquired = self.n_data_not_last * (self.sequential_rounds - 1)
        self.n_data_last_round = self.n_ndata - n_data_acquired

def AcquistionFactory(strategy, sequential_rounds_config):
    if strategy == 'non_seq':
        return NonSeqAcquistion()
    else:
        return SequentialAc(sequential_rounds_config)

class ModelConfig():
    batch_size: int
    class_number: int
    model_dir: str
    def __init__(self, batch_size, class_number, model_dir,device) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.model_dir = model_dir
        self.root = os.path.join(config['base_root'],model_dir,str(batch_size))
        self.check_dir(self.root)
        self.device = device
    def check_dir(self, dir):
        if os.path.exists(dir) is False:
            os.makedirs(dir)
class OldModel(ModelConfig):
    path: str
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt) -> None:
        super().__init__(batch_size, class_number, model_dir, device)
        self.path = os.path.join(self.root,'{}.pt'.format(model_cnt))

class NewModel(ModelConfig):
    path: str
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, pure:bool, setter, augment:bool) -> None:
        super().__init__(batch_size, class_number, model_dir, device)
        self.pure = pure
        self.setter = setter
        self.model_cnt = model_cnt
        self.augment = augment
        self.set_root()

    def set_path(self,acquistion_config:Acquistion):
        if 'seq' in acquistion_config.method:
            root = self.set_seq_root(self.root, acquistion_config)
        else:
            root = self.root
        self.path = os.path.join(root, '{}_{}.pt'.format(acquistion_config.method, acquistion_config.n_ndata))
    
    def set_root(self):
        pure_name = 'pure' if self.pure else 'non-pure'
        aug_name = '' if self.augment else 'na'
        self.root = os.path.join(self.root, self.setter, pure_name, str(self.model_cnt), aug_name) 
        # self.root = os.path.join(self.root, self.setter, pure_name, str(self.model_cnt), aug_name, 'trial') 
        self.check_dir(self.root)

    def set_seq_root(self,root, acquistion_config:SequentialAc):
        # root = os.path.join(root,'{}_rounds'.format(acquistion_config.sequential_rounds_info[acquistion_config.n_ndata]))
        self.check_dir(root)
        return root

class Log(NewModel):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, pure, setter, augment, log_symbol) -> None:
        super().__init__(batch_size, class_number, model_dir, device, model_cnt, pure, setter, augment)
        self.set_log_root(log_symbol)
    def set_log_root(self, log_symbol):
        '''
        Add sub_log symbol ('data','indices',...) to the root
        '''
        self.root = os.path.join(self.root,'log', log_symbol)
        self.check_dir(self.root)
        self.log_symbol = log_symbol

def str2bool(value):
    if isinstance(value,bool):
        return value
    else:
        return False if value=='0' else True

def parse(pure:bool):
    pure_name = 'pure' if pure else 'non-pure'
    hparams = config['hparams']
    batch_size = hparams['batch_size']
    data_config = config['data']
    label_map = data_config['label_map']
    n_new_data = data_config['n_new_data']
    img_per_cls_list = n_new_data
    superclass_num = 2
    ratio = data_config['ratio']
    seq_rounds = 2
    train_labels = data_config['train_label']
    output = {
        'batch_size': batch_size,
        'label_map': label_map,
        'n_data_per_cls': img_per_cls_list,
        'ratio': ratio,
        'removed_labels': config['data']['remove_fine_labels']
    }
    print(output)
    return batch_size, train_labels, label_map, img_per_cls_list, superclass_num, ratio, seq_rounds 