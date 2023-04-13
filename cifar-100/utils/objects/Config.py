import os
from utils import config
from abc import abstractmethod
import re

class Acquistion():
    method: str
    new_data_number_per_class:int

    def __init__(self) -> None:
        self.method = ''
        self.new_data_number_per_class = 0

    def set_items(self, method, new_data_number):
        self.method = method
        self.new_data_number_per_class = new_data_number

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
        return class_number*self.new_data_number_per_class
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{}'.format(self.method, self.new_data_number_per_class)

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
        self.sequential_rounds = self.sequential_rounds_info[self.new_data_number_per_class]
        self.n_data_not_last = self.new_data_number_per_class  // self.sequential_rounds
        n_data_acquired = self.n_data_not_last * (self.sequential_rounds - 1)
        self.n_data_last_round = self.new_data_number_per_class - n_data_acquired

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
        self.path = os.path.join(root, '{}_{}.pt'.format(acquistion_config.method, acquistion_config.new_data_number_per_class))
    
    def set_root(self):
        pure_name = 'pure' if self.pure else 'non-pure'
        aug_name = '' if self.augment else 'na'
        self.root = os.path.join(self.root, self.setter, pure_name, str(self.model_cnt), aug_name) 
        # self.root = os.path.join(self.root, self.setter, pure_name, str(self.model_cnt), aug_name, 'trial') 
        # self.root = os.path.join(self.root, self.setter, pure_name, str(self.model_cnt), aug_name, 'trial_init') 
        self.check_dir(self.root)

    def set_seq_root(self,root, acquistion_config:SequentialAc):
        root = os.path.join(root,'{}_rounds'.format(acquistion_config.sequential_rounds_info[acquistion_config.new_data_number_per_class]))
        self.check_dir(root)
        return root

class Log(NewModel):
    def __init__(self, batch_size, class_number, model_dir, device, model_cnt, pure, setter, augment) -> None:
        super().__init__(batch_size, class_number, model_dir, device, model_cnt, pure, setter, augment)
        self.set_log_root()
    def set_log_root(self):
        self.root = os.path.join(self.root,'log')
        self.check_dir(self.root)
    def set_sub_log_root(self, symbol):
        '''
        Append sub_log symbol ('data','indices',...) to the root
        '''
        self.root = os.path.join(self.root, symbol)
        self.sub_log_symbol = symbol

def str2bool(value):
    if isinstance(value,bool):
        return value
    else:
        return False if value=='0' else True

def parse(model_dir, pure:bool):
    pure_name = 'pure' if pure else 'non-pure'
    hparams = config['hparams']
    batch_size = hparams['batch_size'][model_dir]
    data_config = config['data']
    select_fine_labels = data_config['selected_labels'][re.findall("\d+-class-?[mini]*",model_dir)[0]]
    label_map = data_config['label_map'][re.findall("\d+-class",model_dir)[0]]
    acquired_num_per_class = data_config['acquired_num_per_class'][pure_name]
    img_per_cls_list = acquired_num_per_class['mini'] if 'mini' in model_dir else acquired_num_per_class['non-mini']
    superclass_num = int(model_dir.split('-')[0])
    ratio_symbol = re.sub("\d+-class-?", '', model_dir)
    ratio_key = 'non-mini' if ratio_symbol == '' else ratio_symbol
    ratio = data_config['ratio'][ratio_key]
    seq_rounds = config['seq_rounds']
    return batch_size, select_fine_labels, label_map, img_per_cls_list, superclass_num, ratio, seq_rounds

def display(batch_size, select_fine_labels, label_map, img_per_cls_list, superclass_num, ratio):
    output = {
        'batch_size': batch_size,
        'select_fine_labels': select_fine_labels,
        'label_map': label_map,
        'n_data_per_cls': img_per_cls_list,
        'superclass_num': superclass_num,
        'ratio': ratio
    }
    print(output)