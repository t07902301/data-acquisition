from utils.set_up import *
from utils.logging import *
from utils.env import *
import utils.objects.model as Model
from utils.objects.log import Log
from utils.parse_args import ParseArgs
import utils.objects.Detector as Detector

class Builder():
    def __init__(self, dataset_name, data_config, normalize_stat) -> None:
        if dataset_name == 'cifar':
            self.data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            self.data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])
    
    def set_validation(self, new_batch_size, val_shift_data):
        self.validation_loader = torch.utils.data.DataLoader(val_shift_data, batch_size=new_batch_size)
        return
    
    def run(self, epochs, parse_args:ParseArgs, method_list, budget_list, operation:Config.Operation, dataset_list):
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))

            _, new_model_config, general_config = parse_args.get_model_config(epo)

            self.set_validation(new_model_config.new_batch_size, dataset_list[epo]['val_shift'])
            self.epoch_run(method_list, budget_list, operation, new_model_config, general_config)
    
    def epoch_run(self, method_list, budget_list, operation:Config.Operation, new_model_config:Config.NewModel, general_config):
        for method in method_list:
            operation.acquisition.set_method(method)
            self.method_run(budget_list, operation, new_model_config, general_config)

    def method_run(self, budget_list, operation:Config.Operation, new_model_config:Config.NewModel, general_config):
        for budget in budget_list:
            operation.acquisition.set_budget(budget)
            new_model_config.set_path(operation)
            self.budget_run(operation, new_model_config, general_config)
    
    def get_train_loader(self, operation:Config.Operation, general_config, new_model_config:Config.NewModel):
        log = Log(new_model_config, 'indices')
        new_data_indices = log.import_log(operation, general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices)   
        return new_data
    
    def build_padding(self, new_model_config: Config.NewModel, train_data, val_loader, config, vit=None):
        '''
        Update model after training set in workspace is refreshed
        '''
        if new_model_config.model_type == 'cnn':
            padding = Model.CNN(config)
            generator = dataloader_env()
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=new_model_config.new_batch_size, shuffle=True, drop_last=True, generator=generator)
            padding.train(train_loader, val_loader, config)
        else:
            padding = Model.svm(config['detector_args'], vit)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=new_model_config.new_batch_size)
            padding.train(train_loader) 
        padding.save(new_model_config.path)
        
    
    def budget_run(self, operation:Config.Operation, new_model_config:Config.NewModel, general_config):
        train_loader = self.get_train_loader(operation, general_config, new_model_config)
        self.build_padding(new_model_config, train_loader, self.validation_loader, general_config['hparams']['padding'], operation.detection.vit)

def main(epochs,  model_dir ='', device=0, acquisition_method= 'all', detector='svm'):
    
    methold_list = ['conf', 'mix'] if acquisition_method == 'all' else [acquisition_method] 
    # methold_list = [acquisition_method]
    fh = logging.FileHandler('log/{}/{}.log'.format(model_dir, acquisition_method), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)

    ensemble_instruction = Config.Ensemble()
    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    detect_instruction = Config.Detection(detector, vit_mounted=clip_processor, weaness_label_generator=None)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator='u-ws')
    
    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    
    builder = Builder(parse_args.dataset_name, parse_args.general_config['data'], normalize_stat)
    builder.run(epochs, parse_args, methold_list, parse_args.general_config['data']['budget'], operation, dataset_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device,  acquisition_method=args.acquisition_method, detector=args.detector_name)