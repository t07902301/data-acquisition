from utils.strategy import *
import utils.statistics.checker as Checker
from utils.set_up import *

def method_run(budget_list, operation:Config.Operation, checker: Checker.Prototype):
    acc_change_list = []
    for budget in budget_list:
        operation.acquisition.set_budget(budget)
        acc_change = budget_run(operation, checker)
        acc_change_list.append(acc_change)
    return acc_change_list

def budget_run(operation:Config.Operation, checker: Checker.Prototype):
    check_result = checker.run(operation)
    return check_result

def run(operation:Config.Operation, budget_list, checker:Checker.Prototype):
    logger.info('In method: {}'.format(operation.acquisition.method))
    result = method_run(budget_list, operation, checker)
    return result

# def check_threshold(old_model_config: Config.OldModel, datasplit: Dataset.DataSplits, acquire_instruction: Config.Acquisition, clip_processor):
#     base_model = Model.resnet(2)
#     base_model.load(old_model_config)
#     clf = Detector.SVM(datasplit.loader['train_clip'], clip_processor)
#     _ = clf.fit(base_model, datasplit.loader['val_shift'])
#     market_dv, _ = clf.predict(datasplit.loader['market'])
#     return (market_dv <= acquire_instruction.threshold).sum()

def epoch_run(budget_list, operation: Config.Operation, checker: Checker.Prototype):
    result_epoch = run(operation, budget_list, checker)
    # threshold_stat = check_threshold(old_model_config, dataset_splits,acquire_instruction, clip_processor)
    return result_epoch, 0

def main(epochs, acquisition_method, device, detector_name, model_dir, ensemble_name, base_type, ensemble_criterion, pdf= 'kde'):
    fh = logging.FileHandler('log/{}/test_{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    new_model_setter = 'retrain'
    pure = True
    
    if acquisition_method == 'rs':
        ensemble_criterion, ensemble_name = 0, 'wta'

    logger.info('Ensemble Name: {}, criterion:{}'.format(ensemble_name, ensemble_criterion))

    config, device_config, dataset_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(ensemble_criterion, ensemble_name, pdf)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'])

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)

    results = []
    threshold_stat_list = []
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation) #probab / ensemble
        result_epoch, threshold_stat = epoch_run(config['data']['budget'], operation, checker)
        results.append(result_epoch)
        threshold_stat_list.append(threshold_stat)

    result = np.array(result)
    logger.info(acquisition_method)
    logger.info('avg:{}'.format(np.round(np.mean(result, axis=0), decimals=3).tolist()))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-ec','--criterion',type=float,default=0.5, help='Criterion in WTA ensemble')
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv: u-wfs, rs: random, conf: confiden-score, seq: sequential u-wfs, pd: u-wfsd, seq_pd: sequential u-wfsd")
    parser.add_argument('-em','--ensemble',type=str, default='ae', help="Ensemble Method")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    args = parser.parse_args()

    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, ensemble_name=args.ensemble, base_type=args.base_type, ensemble_criterion=args.criterion)
