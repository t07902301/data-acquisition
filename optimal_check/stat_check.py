import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))
from utils.strategy import *
from utils.set_up import *
from utils.logging import *

class TrainData():
    def __init__(self, dataset_name, data_config, normalize_stat) -> None:
        if dataset_name == 'cifar':
            self.data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            self.data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])

    def run(self, epochs, parse_args:ParseArgs, budget_list, operation:Config.Operation, general_config):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            source_model_config, new_model_config, general_config = parse_args.get_model_config(epo)

            optimal_config = parse_args.general_config['hparams']['optimal']
            opt_model_config = Config.OptModel(optimal_config['batch_size']['base'], optimal_config['superclass'], parse_args.model_dir, parse_args.device_config, epo, parse_args.general_config['base_type'])
            opt_model = Model.CNN(optimal_config)
            opt_model.load(opt_model_config.path, opt_model_config.device)

            # source_model = Model.factory(source_model_config.base_type, parse_args.general_config, source=True)
            # source_model.load(source_model_config.path, source_model_config.device)

            result_epoch = self.epoch_run(operation, budget_list, new_model_config, general_config, opt_model, source_model=None)
            results.append(result_epoch)
            
            del opt_model
        return results

    def epoch_run(self, operation:Config.Operation, budget_list, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, source_model: Model.Prototype):
        return self.method_run(budget_list, operation, new_model_config, general_config, opt_model, source_model)

    def method_run(self, budget_list, operation:Config.Operation, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, source_model: Model.Prototype):
        acc_change_list = []
        for budget in budget_list:
            operation.acquisition.set_budget(budget)
            acc_change = self.budget_run(operation, new_model_config, general_config, opt_model, source_model)
            acc_change_list.append(acc_change)
        return acc_change_list
    
    def get_invalid_size(self, opt_model: Model.Prototype, batch_size, data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        return 100 - opt_model.acc(data_loader)
    
        # return (gts!=preds).sum()
        # invalid_indices = np.arange(len(data))[gts!=preds]
        # invalid = torch.utils.data.Subset(data, invalid_indices)
        # return invalid
    
    def get_source_mistakes(self, source_model:Model.Prototype, batch_size, data):
        # PGT and random weakess always return 100% weaknesses
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        return 100 - source_model.acc(data_loader)
    
    def check_indices(self, operation:Config.Operation, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, source_model: Model.Prototype):
        new_model_config.set_path(operation)
        log = Log(new_model_config, 'indices')
        new_data_indices = log.import_log(operation, general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices)
        return self.get_invalid_size(opt_model, new_model_config.new_batch_size, new_data)
        # return self.get_source_mistakes(source_model, new_model_config.new_batch_size, new_data)
        # return self.get_valid_mistakes(opt_model, new_model_config.new_batch_size, mistakes)

    def budget_run(self, operation:Config.Operation, new_model_config, general_config, opt_model:Model.Prototype, source_model: Model.Prototype):
        check_result = self.check_indices(operation, new_model_config, general_config, opt_model, source_model)
        return check_result
   
def main(epochs, model_dir ='', device=0,acquisition_method= 'dv', detector_name=''):
    fh = logging.FileHandler('log/{}/stat_{}_invalid.log'.format(model_dir, acquisition_method), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    parse_args, _, normalize_stat = set_up(epochs, model_dir, device)

    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, None, None)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=None)   
    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    
    stat_checker = TrainData(parse_args.dataset_name, parse_args.general_config['data'], normalize_stat)

    results = stat_checker.run(epochs, parse_args, parse_args.general_config['data']['budget'], operation, parse_args.general_config)
    results = np.array(results)
    logger.info('Train Data stat: {}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))
    # logger.info('all: {}'.format(results.tolist()))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv: u-wfs, rs: random, conf: confiden-score, seq: sequential u-wfs, pd: u-wfsd, seq_pd: sequential u-wfsd")
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, acquisition_method=args.acquisition_method, detector_name=args.detector_name)
