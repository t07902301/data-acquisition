
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