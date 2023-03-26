import os
from utils import config
from abc import abstractmethod

class AcquistionConfig():
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

class NonSeqAcquistionConfig(AcquistionConfig):
    def __init__(self) -> None:
        super().__init__()

    def get_new_data_size(self, class_number):
        return class_number*self.new_data_number_per_class
    
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{}'.format(self.method, self.new_data_number_per_class)

class SequentialAcConfig(AcquistionConfig):
    def __init__(self, sequential_rounds:int) -> None:
        super().__init__()
        self.sequential_rounds = sequential_rounds
        self.round_acquire_method = 'dv'
        self.current_round = 0
    def set_round(self, round):
        self.current_round = round + 1
    def get_new_data_size(self, class_number):
        return class_number*self.round_data_per_class*self.sequential_rounds
    def get_info(self):
        return 'acquisition method: {}, n_data_per_class:{} in round {}'.format(self.method, self.round_data_per_class, self.current_round)

    def set_items(self, method, new_data_number):
        super().set_items(method, new_data_number)
        self.round_data_per_class = self.new_data_number_per_class  // self.sequential_rounds

def AcquistionConfigFactory(method, sequential_rounds):
    if method == 'non_seq':
        return NonSeqAcquistionConfig()
    else:
        return SequentialAcConfig(sequential_rounds)

class ModelConfig():
    batch_size: int
    class_number: int
    model_dir: str
    def __init__(self, batch_size, class_number, model_dir) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.model_dir = model_dir
        self.root = os.path.join(config['base_root'],model_dir,str(batch_size))
        self.check_dir(self.root)
    def check_dir(self, dir):
        if os.path.exists(dir) is False:
            os.makedirs(dir)
class OldModelConfig(ModelConfig):
    path: str
    def __init__(self, batch_size, class_number, model_dir, model_cnt) -> None:
        super().__init__(batch_size, class_number, model_dir)
        self.path = os.path.join(self.root,'{}.pt'.format(model_cnt))
class NewModelConfig(ModelConfig):
    def __init__(self, batch_size, class_number, model_dir, model_cnt, pure, setter) -> None:
        super().__init__(batch_size, class_number, model_dir)
        self.pure = pure
        self.setter = setter
        self.path = ''
        self.model_cnt = model_cnt
        self.set_root(setter)
    def set_path(self,acquistion_config:AcquistionConfig):
        self.path = os.path.join(self.root, '{}_{}.pt'.format(acquistion_config.method, acquistion_config.new_data_number_per_class))
    def set_root(self,setter):
        pure_name = 'pure' if self.pure else 'non-pure'
        self.root = os.path.join(self.root,setter,pure_name, str(self.model_cnt)) 
        self.check_dir(self.root)

class LogConfig(NewModelConfig):
    def __init__(self, batch_size, class_number, model_dir, model_cnt, pure, setter) -> None:
        super().__init__(batch_size, class_number, model_dir, model_cnt, pure, setter)
        self.set_log_root()
    def set_log_root(self):
        self.root = os.path.join(self.root,'log')
        self.check_dir(self.root)

def parse_config(model_dir, pure:bool):
    pure_name = 'pure' if pure else 'non-pure'
    hparams = config['hparams']
    batch_size = hparams['batch_size'][model_dir]
    data_config = config['data']
    select_fine_labels = data_config['selected_labels'][model_dir]
    label_map = data_config['label_map'][model_dir]   
    acquired_num_per_class = data_config['acquired_num_per_class'][pure_name]
    img_per_cls_list = acquired_num_per_class['mini'] if 'mini' in model_dir else acquired_num_per_class['non-mini']
    superclass_num = int(model_dir.split('-')[0])
    return batch_size, select_fine_labels, label_map, img_per_cls_list, superclass_num
