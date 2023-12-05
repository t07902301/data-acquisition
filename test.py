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
    budget_result = checker.run(operation)
    return budget_result

def run(operation:Config.Operation, budget_list, checker:Checker.Prototype):
    logger.info('In method: {}'.format(operation.acquisition.method))
    result = method_run(budget_list, operation, checker)
    return result

def epoch_run(budget_list, operation: Config.Operation, checker: Checker.Prototype):
    result_epoch = run(operation, budget_list, checker)
    return result_epoch

def main(epochs, acquisition_method, device, detector_name, model_dir, ensemble_name, base_type, ensemble_criterion, utility_estimator, use_posterior):
    fh = logging.FileHandler('log/{}/test_{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    new_model_setter = 'retrain'
    pure = True
    
    if acquisition_method == 'rs':
        ensemble_name = 'total'

    if use_posterior is False:
        assert detector_name == 'logregs'

    logger.info('Ensemble Name: {}, criterion:{}, use_posterior: {}, utility_estimator: {}'.format(ensemble_name, ensemble_criterion, use_posterior, utility_estimator))

    config, device_config, dataset_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(name=ensemble_name, criterion=ensemble_criterion)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)

    results = []
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, normalize_stat, dataset_name, use_posterior) #probab / ensemble
        result_epoch = epoch_run(config['data']['budget'], operation, checker)
        results.append(result_epoch)

    results = np.array(results)
    logger.info(acquisition_method)
    logger.info('avg:{}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-ec','--criterion',type=float,default=0.5, help='Criterion in WTA ensemble')
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: confiden-score, seq: sequential u-ws")
    parser.add_argument('-em','--ensemble',type=str, default='total', help="Ensemble Method")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")
    parser.add_argument('-up','--use_posterior',type=str2bool, default=1, help="use posterior or not")

    args = parser.parse_args()

    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, ensemble_name=args.ensemble, base_type=args.base_type, ensemble_criterion=args.criterion, utility_estimator=args.utility_estimator, use_posterior=args.use_posterior)
