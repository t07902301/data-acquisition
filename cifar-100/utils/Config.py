import os
class ModelConfig():
    batch_size: int
    class_number: int
    model_dir: str
    def __init__(self, batch_size, class_number, model_dir) -> None:
        self.batch_size = batch_size
        self.class_number = class_number
        self.root = os.path.join('new_model',model_dir,str(batch_size))
        self.check_dir()
    def check_dir(self):
        if os.path.exists(self.root) is False:
            os.makedirs(self.root)
class OldModelConfig(ModelConfig):
    path: str
    def __init__(self, batch_size, class_number, model_dir, model_cnt) -> None:
        super().__init__(batch_size, class_number, model_dir)
        self.path = os.path.join(self.root,'{}.pt'.format(model_cnt))
class NewModelConfig(ModelConfig):
    def __init__(self, batch_size, class_number, model_dir, pure, setter) -> None:
        super().__init__(batch_size, class_number, model_dir)
        self.pure = pure
        self.setter = setter
        pure_name = 'pure' if self.pure else ''
        self.root = os.path.join(self.root,setter,pure_name) 
        self.path = ''
        self.check_dir()
    def set_path(self,acquistion_config):
        self.path = os.path.join(self.root,'{}_{}_{}.pt'.format(acquistion_config.method,acquistion_config.new_data_number_per_class,acquistion_config.model_cnt))
class AcquistionConfig():
    def __init__(self, model_cnt, sequential_rounds) -> None:
        self.method = ''
        self.new_data_number_per_class = 0
        self.model_cnt = model_cnt
        self.sequential_rounds = sequential_rounds
        self.round_acquire_method = 'dv'
    def set_items(self, method, new_data_number):
        self.method = method
        self.new_data_number_per_class = new_data_number
# class SequentialAcConfig(AcquistionConfig):
#     def __init__(self, model_cnt, sequential_rounds) -> None:
#         super().__init__(model_cnt)


         