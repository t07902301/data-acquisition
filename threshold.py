'''
This script is used to train models with different utility thresholds and test the performance of the models on the validation set.
Usage: python threshold.py -e 1 -md 'core_object' -d 0 -dn 'svm' -am 'dv' -mode 'train'
'''
from utils.strategy import *
from utils.set_up import *
from utils.logging import *
import utils.statistics.checker as Checker
import utils.dataset.wrappers as Dataset
from typing import List
import os
from utils.parse_args import ParseArgs

class ModelBuilder():
    def __init__(self, dataset_name, data_config, normalize_stat, threshold_list) -> None:
        if dataset_name == 'cifar':
            self.data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            self.data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])
        self.threshold_list = threshold_list
    
    def set_validation(self, new_batch_size, val_shift_data):
        self.validation_loader = torch.utils.data.DataLoader(val_shift_data, batch_size=new_batch_size)
        return
    
    def run(self, epochs, parse_args:ParseArgs, operation:Config.Operation, dataset_list):
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            _, new_model_config, general_config = parse_args.get_model_config(epo)
            self.set_validation(new_model_config.new_batch_size, dataset_list[epo]['val_shift'])
            self.threshold_run(operation, new_model_config, general_config)
    
    def threshold_run(self, operation: Config.Operation, new_model_config:Config.NewModel, general_config):
        operation.acquisition.set_anchor_threshold(np.max(self.threshold_list))
        for threshold in self.threshold_list:
            operation.acquisition.set_threshold(threshold)
            self.build(operation, general_config, new_model_config)

    def build(self, operation:Config.Operation, general_config, new_model_config:Config.NewModel):
        train_loader = self.get_train_data(operation, general_config, new_model_config)
        self.build_padding(new_model_config, train_loader, self.validation_loader, general_config['hparams']['padding'])

    def get_train_data(self, operation:Config.Operation, general_config, new_model_config:Config.NewModel):
        model_config = new_model_config
        model_config.set_path(operation)

        log = Log(model_config, 'indices')
        new_data_indices = log.import_log(operation, general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices) 

        return new_data
    
    def build_padding(self, new_model_config: Config.NewModel, train_data, val_loader, general_config):
        '''
        Update model after training set in workspace is refreshed
        '''
        padding = Model.CNN(general_config)
        generator = dataloader_env()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=new_model_config.new_batch_size, shuffle=True, drop_last=True, generator=generator)
        if len(train_loader) == 0:
            return
        padding.train(train_loader, val_loader, general_config)
        padding.save(new_model_config.path)

class DataCollector():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def run(self, epochs, dataset_list, parse_args:ParseArgs, operation):
        for epo in range(epochs):
            logger.info('In epoch {}'.format(epo))
            dataset = dataset_list[epo]
            self.epoch_run(parse_args, dataset, epo, operation)

    def epoch_run(self, parse_args:ParseArgs, dataset:dict, epo, operation: Config.Operation):
        old_model_config, new_model_config, general_config = parse_args.get_model_config(epo)
        workspace = WorkSpace(old_model_config, dataset, general_config)

        logger.info('Set up WorkSpace')
        
        workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)
        workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)

        self.method_run(new_model_config, operation, workspace)

    def method_run(self, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
        # method = operation.acquisition.method
        # if method != 'rs':
        #     workspace.set_validation(new_model_config.new_batch_size)
        #     workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)
        # else:
        # workspace.validation_loader = workspace.data_split.loader['val_shift']
        # logger.info('Keep val_shift for validation_loader') # Align with inference on the test set

        self.threshold_run(operation, new_model_config, workspace)

    def threshold_run(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
        operation.acquisition.set_anchor_threshold(np.max(self.threshold_list))
        for threshold in self.threshold_list:
            operation.acquisition.set_threshold(threshold)
            self.acquire(operation, new_model_config, workspace)

    def acquire(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
        strategy = StrategyFactory(operation.acquisition.method)
        strategy.operate(operation, new_model_config, workspace)

class Test():
    def __init__(self, threshold_list) -> None:
        self.threshold_list = threshold_list

    def threshold_run(self, operation: Config.Operation, checker: Checker.Prototype):
        acc_change_list = []
        for threshold in self.threshold_list:
            operation.acquisition.threshold = threshold
            checker.new_model_config.set_path(operation)
            if os.path.exists(checker.new_model_config.path) is False:
                continue
            acc_change = checker.run(operation)
            acc_change_list.append(acc_change)
            # logger.info(acc_change)
        return acc_change_list

    def epoch_run(self, operation: Config.Operation, checker: Checker.Prototype):
        logger.info('In method: {}'.format(operation.acquisition.method))
        acc_change = self.threshold_run(operation, checker)
        return acc_change

    def run(self, epochs, parse_args:ParseArgs, dataset_list, operation: Config.Operation, normalize_stat, use_posterior):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, normalize_stat, parse_args.dataset_name, use_posterior) #probab / ensemble
            result_epoch = self.epoch_run(operation, checker)
            if result_epoch == []:
                continue
            results.append(result_epoch)
        logger.info('avg:{}'.format(np.round(np.mean(results, axis=0), decimals=3)))

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

    def run(self, epochs, parse_args:ParseArgs, dataset_list, operation: Config.Operation, normalize_stat, data_config):
        results = []
        if parse_args.dataset_name == 'cifar':
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
    
    def threshold_run(self, threshold_list, operation: Config.Operation, new_model_config:Config.NewModel, general_config, dataset_splits, weakness):
        results = []
        for threshold in threshold_list:
            operation.acquisition.set_threshold(threshold)
            new_model_config.set_path(operation)
            model = Model.factory(new_model_config.model_type, general_config, source=False)
            model.load(new_model_config.path, new_model_config.device)
            gts, preds, decision_scores = model.eval(dataset_splits.loader['val_shift'])
            targets = (gts!=preds) if weakness else (gts==preds)
            results.append(np.mean(acquistion.get_gt_probab(gts[targets], decision_scores[targets])))
        return results

    def run(self, epochs, parse_args:ParseArgs, threshold_list, operation:Config.Operation, dataset_list: List[dict], weakness):
        results = []
        for epo in range(epochs):
            _, new_model_config, general_config = parse_args.get_model_config(epo)
            logger.info('in epoch {}'.format(epo))
            dataset_splits = Dataset.DataSplits(dataset_list[epo], new_model_config.new_batch_size)
            threshold_result = self.threshold_run(threshold_list, operation, new_model_config, general_config, dataset_splits, weakness)
            results.append(threshold_result)
        
        results = np.round(np.mean(results, axis=0), decimals=3)
        results = {threshold_list[idx]:results[idx]  
                   for idx in range(len(threshold_list))}

        logger.info('Model Confidence stat:{}'.format(results))

def main(epochs, acquisition_method, device, detector_name, model_dir, mode, ensemble_name, utility_estimator, use_posterior, ensemble_criterion):
    threshold_list = [0.5, 0.6, 0.7]

    fh = logging.FileHandler('log/{}/threshold_{}.log'.format(model_dir, mode),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Ensemble Name: {}, criterion:{}, use_posterior: {}, utility_estimator: {}'.format(ensemble_name, ensemble_criterion, use_posterior, utility_estimator))

    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(name=ensemble_name, criterion=ensemble_criterion)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    
    if mode == 'acquire':
        DataCollector(threshold_list).run(epochs, dataset_list, parse_args, operation)

    elif mode == 'train':
        ModelBuilder(parse_args.dataset_name, parse_args.general_config['data'], normalize_stat, threshold_list).run(epochs, parse_args, operation, dataset_list)

    elif mode == 'stat':
        Stat(threshold_list).run(epochs, parse_args, dataset_list, operation, normalize_stat, parse_args.general_config['data'])

    elif mode == 'conf':
        ModelConf().run(epochs, parse_args, threshold_list, operation, dataset_list, weakness=False)

    else:
        Test(threshold_list).run(epochs, parse_args, dataset_list, operation, normalize_stat, use_posterior)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")
    parser.add_argument('-mode','--mode',type=str,default='test', help="train or test models from utility thresholds")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")
    parser.add_argument('-up','--use_posterior',type=str2bool, default=1, help="use posterior or not")
    parser.add_argument('-em','--ensemble',type=str, default='ae-w', help="Ensemble Method")
    parser.add_argument('-ec','--ensemble_criterion',type=float,default=0.5, help='A threshold of the probability from Cw to assign test set and create corresponding val set for model training.')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, mode=args.mode, ensemble_name=args.ensemble, utility_estimator=args.utility_estimator, use_posterior=args.use_posterior, ensemble_criterion=args.ensemble_criterion)