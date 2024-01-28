from utils.objects.Config import OldModel, NewModel

class ParseArgs():
    def __init__(self, model_dir=None, device_config=None, config=None, dataset_name=None, task=None, filter_market=None) -> None:
        self.model_dir = model_dir
        self.device_config = device_config
        self.pure = True
        self.new_model_setter = 'retrain'
        self.general_config = config
        self.filter_market = filter_market
        self.dataset_name = dataset_name
        self.task = task
        self.filter_market = filter_market

    def get_model_config(self, epoch):
        batch_size = self.general_config['hparams']['padding']['batch_size']
        superclass_num = self.general_config['hparams']['padding']['superclass']
        old_model_config = OldModel(batch_size['base'], superclass_num, self.model_dir, self.device_config, epoch, base_type=self.general_config['base_type'])
        new_model_config = NewModel(batch_size['base'], superclass_num, self.model_dir, self.device_config, epoch, self.pure, self.new_model_setter, batch_size['new'], base_type=self.general_config['base_type'], padding_type=self.general_config['padding_type'])
        return old_model_config, new_model_config, self.general_config