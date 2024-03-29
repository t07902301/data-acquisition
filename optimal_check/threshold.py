import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))
from utils.strategy import *
from utils.set_up import *
from utils.logging import *

class Stat():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def get_intractable(self, opt_model: Model.Prototype, batch_size, data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        gts, preds, _ = opt_model.eval(data_loader)
        intractable_indices = np.arange(len(data))[gts!=preds]
        intractable = torch.utils.data.Subset(data, intractable_indices)
        return intractable
    
    def check_indices(self, operation:Config.Operation, data, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, old_model: Model.Prototype):
        new_model_config.set_path(operation)
        log = Log(new_model_config, 'indices')
        new_data_indices = log.import_log(operation, general_config)
        new_data = torch.utils.data.Subset(data, new_data_indices)
        return len(self.get_intractable(opt_model, new_model_config.new_batch_size, new_data)) / len(new_data) * 100
        # return len(self.get_intractable(opt_model, new_model_config.new_batch_size, new_data))

    def threshold_run(self, operation: Config.Operation, data, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, old_model: Model.Prototype):
        acc_change_list = []
        for threshold in self.threshold_list:
            operation.acquisition.threshold = threshold
            acc_change = self.check_indices(operation, data, new_model_config, general_config, opt_model, old_model)
            acc_change_list.append(acc_change)
        return acc_change_list

    def epoch_run(self, operation:Config.Operation, data, new_model_config:Config.NewModel, general_config, opt_model:Model.Prototype, old_model: Model.Prototype):
        logger.info('In method: {}'.format(operation.acquisition.method))
        acc_change = self.threshold_run(operation, data, new_model_config, general_config, opt_model, old_model)
        return acc_change

    def run(self, epochs, parse_args:ParseArgs, operation: Config.Operation, dataset_name, normalize_stat, data_config):
        results = []
        if dataset_name == 'cifar':
            data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])

        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            source_model_config, new_model_config, config = parse_args.get_model_config(epo)
            
            optimal_config = parse_args.general_config['hparams']['optimal']
            opt_model_config = Config.OptModel(optimal_config['batch_size']['base'], optimal_config['superclass'], parse_args.model_dir, parse_args.device_config, epo, parse_args.general_config['base_type'])
            opt_model = Model.CNN(optimal_config)
            opt_model.load(opt_model_config.path, opt_model_config.device)

            source_model = Model.factory(source_model_config.model_type, config)
            source_model.load(source_model_config.path, source_model_config.device)

            result_epoch = self.epoch_run( operation, data, new_model_config, config, opt_model, source_model)
            results.append(result_epoch)
        logger.info('avg:{}'.format(np.round(np.mean(results, axis=0), decimals=3)))

def main(epochs, acquisition_method, device, detector_name, model_dir):
    threshold_list = [0.5, 0.6, 0.7]

    fh = logging.FileHandler('log/{}/threshold_invalid.log'.format(model_dir),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    parse_args, _, normalize_stat = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=None)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
   
    Stat(threshold_list).run(epochs, parse_args, operation, parse_args.dataset_name, normalize_stat, parse_args.general_config['data'])

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv: u-wfs, rs: random, conf: confiden-score, seq: sequential u-wfs, pd: u-wfsd, seq_pd: sequential u-wfsd")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method)