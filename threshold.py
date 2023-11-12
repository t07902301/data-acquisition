from utils.strategy import *
from utils.set_up import *
from utils.logging import *
import utils.statistics.checker as Checker
import utils.dataset.wrappers as Dataset
from typing import List

class Train():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def run(self, epochs, dataset_list, parse_args, operation):
        for epo in range(epochs):
            logger.info('In epoch {}'.format(epo))
            dataset = dataset_list[epo]
            self.epoch_run(parse_args, dataset, epo, operation)

    def epoch_run(self, parse_args, dataset:dict, epo, operation: Config.Operation):
        model_dir, device_config, base_type, pure, new_model_setter, config = parse_args
        batch_size = config['hparams']['source']['batch_size']
        superclass_num = config['hparams']['source']['superclass']

        old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
        new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
        workspace = WorkSpace(old_model_config, dataset, config)

        logger.info('Set up WorkSpace')
        
        workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)

        self.method_run(new_model_config, operation, workspace)

    def method_run(self, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
        method = operation.acquisition.method
        if method != 'rs':
            workspace.set_detector(operation.detection)
            workspace.set_validation(new_model_config.new_batch_size)
            if 'pd' in method:
                workspace.set_anchor_dstr(operation.ensemble.pdf)
        
        else:
            workspace.validation_loader = workspace.data_split.loader['val_shift']
            logger.info('Keep val_shift for validation_loader') # Align with inference on the test set

        self.threshold_run(operation, new_model_config, workspace)

    def threshold_run(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
        operation.acquisition.set_anchor_threshold(np.max(self.threshold_list))
        for threshold in self.threshold_list:
            operation.acquisition.set_threshold(threshold)
            self.train(operation, new_model_config, workspace)

    def train(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
        strategy = StrategyFactory(operation.acquisition.method)
        strategy.operate(operation, new_model_config, workspace)

class Test():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def threshold_run(self, operation: Config.Operation, checker: Checker.Prototype):
        acc_change_list = []
        for threshold in self.threshold_list:
            operation.acquisition.threshold = threshold
            acc_change = checker.run(operation)
            acc_change_list.append(acc_change)
        return acc_change_list

    def epoch_run(self, operation: Config.Operation, checker: Checker.Prototype):
        logger.info('In method: {}'.format(operation.acquisition.method))
        acc_change = self.threshold_run(operation, checker)
        return acc_change

    def run(self, epochs, parse_args, dataset_list, operation: Config.Operation):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation) #probab / ensemble
            result_epoch = self.epoch_run( operation, checker)
            results.append(result_epoch)
        # logger.info('avg:{}'.format(np.round(np.mean(results, axis=0), decimals=3)))

class Stat():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def utility_range(self, dataset, batch_size, detector: Detector.Prototype):
        new_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        utility, _ = detector.predict(new_data_loader)
        return min(utility)

    def misclassifications(self, dataset, batch_size, base_model: Model.Prototype):
        new_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # return (100 - base_model.acc(new_data_loader)) * dataloader_utils.get_size(new_data_loader) /100
        return 100 - base_model.acc(new_data_loader)

    def check_indices(self, operation:Config.Operation, checker:Checker.Prototype, data):
        model_config = checker.new_model_config
        model_config.set_path(operation)

        log = Log(model_config, 'indices')
        new_data_indices = log.import_log(operation, checker.general_config)
        new_data = torch.utils.data.Subset(data, new_data_indices)
        # logger.info(len(new_data))

        # return self.get_dataset_size(new_data)
        # return self.utility_range(new_data, model_config.new_batch_size, checker.detector)
        return self.misclassifications(new_data, model_config.new_batch_size, checker.base_model)

    def threshold_run(self, operation: Config.Operation, checker: Checker.Prototype, data):
        acc_change_list = []
        for threshold in self.threshold_list:
            operation.acquisition.threshold = threshold
            acc_change = self.check_indices(operation, checker, data)
            acc_change_list.append(acc_change)
        return acc_change_list

    def epoch_run(self, operation: Config.Operation, checker: Checker.Prototype, data):
        logger.info('In method: {}'.format(operation.acquisition.method))
        acc_change = self.threshold_run(operation, checker, data)
        return acc_change

    def run(self, epochs, parse_args, dataset_list, operation: Config.Operation, dataset_name, normalize_stat, data_config):
        results = []
        if dataset_name == 'cifar':
            data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])

        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation) #probab / ensemble
            result_epoch = self.epoch_run( operation, checker, data)
            results.append(result_epoch)
        logger.info('avg:{}'.format(np.round(np.mean(results, axis=0), decimals=3)))

class ModelConf():
    def __init__(self) -> None:
        pass
    
    def threshold_run(self, threshold_list, operation: Config.Operation, new_model_config:Config.NewModel, dataset_splits, config, weakness):
        results = []
        for threshold in threshold_list:
            operation.acquisition.set_threshold(threshold)
            new_model_config.set_path(operation)
            model = Model.factory(new_model_config.base_type, config)
            model.load(new_model_config.path, new_model_config.device)
            gts, preds, decision_scores = model.eval(dataset_splits.loader['val_shift'])
            targets = (gts!=preds) if weakness else (gts==preds)
            results.append(np.mean(acquistion.get_gt_probab(gts[targets], decision_scores[targets])))
        return results

    def run(self, epochs, parse_args, threshold_list, operation:Config.Operation, dataset_list: List[dict], weakness):
        results = []
        model_dir, device_config, base_type, pure, new_model_setter, config = parse_args
        batch_size = config['hparams']['source']['batch_size']
        superclass_num = config['hparams']['source']['superclass']
        for epo in range(epochs):
            new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
            logger.info('in epoch {}'.format(epo))
            dataset_splits = Dataset.DataSplits(dataset_list[epo], new_model_config.new_batch_size)
            threshold_result = self.threshold_run(threshold_list, operation, new_model_config, dataset_splits, config, weakness)
            results.append(threshold_result)
        
        results = np.round(np.mean(results, axis=0), decimals=3)
        results = {threshold_list[idx]:results[idx]  
                   for idx in range(len(threshold_list))}

        logger.info('Model Confidence stat:{}'.format(results))

def main(epochs, acquisition_method, device, detector_name, model_dir, base_type, mode):
    threshold_list = [0.4, 0.8]
    # threshold_list = [0.5, 0.6, 0.7]

    fh = logging.FileHandler('log/{}/threshold_{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    pure, new_model_setter = True, 'retrain'

    config, device_config, ds_list, normalize_stat,dataset_name, option = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'])

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
   
    if mode == 'train':
        parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
        Train(threshold_list).run(epochs, ds_list, parse_args, operation)

    elif mode == 'stat':
        parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
        Stat(threshold_list).run(epochs, parse_args, ds_list, operation, dataset_name, normalize_stat, config['data'])

    elif mode == 'conf':
        parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
        ModelConf().run(epochs, parse_args, threshold_list, operation, ds_list, weakness=False)

    else:
        parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
        Test(threshold_list).run(epochs, parse_args, ds_list, operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv: u-wfs, rs: random, conf: confiden-score, seq: sequential u-wfs, pd: u-wfsd, seq_pd: sequential u-wfsd")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-mode','--mode',type=str,default='test', help="train or test models from utility thresholds")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, base_type=args.base_type, mode=args.mode)