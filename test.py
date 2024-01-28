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

def main(epochs, acquisition_method, device, detector_name, model_dir, ensemble_name, ensemble_criterion, utility_estimator, use_posterior, weaness_label_generator):
    fh = logging.FileHandler('log/{}/test_{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    if use_posterior is False:
        assert detector_name == 'logregs'

    logger.info('Ensemble Name: {}, criterion:{}, use_posterior: {}, utility_estimator: {}'.format(ensemble_name, ensemble_criterion, use_posterior, utility_estimator))

    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(name=ensemble_name, criterion=ensemble_criterion)
    detect_instruction = Config.Detection(detector_name, clip_processor, weaness_label_generator)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    results = []
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, normalize_stat, parse_args.dataset_name, use_posterior) #probab / ensemble
        result_epoch = epoch_run(parse_args.general_config['data']['budget'], operation, checker)
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
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")
    parser.add_argument('-em','--ensemble',type=str, default='total', help="Ensemble Method")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")
    parser.add_argument('-up','--use_posterior',type=str2bool, default=1, help="use posterior or not")
    parser.add_argument('-wc','--weaness_label_generator',type=str,default='correctness', help="correctness, entropy+gmm")

    args = parser.parse_args()

    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, ensemble_name=args.ensemble, ensemble_criterion=args.criterion, utility_estimator=args.utility_estimator, use_posterior=args.use_posterior, weaness_label_generator=args.weaness_label_generator)
