from utils.strategy import *
from utils.set_up import *
from utils.logging import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.seq_stat_mode = True
    stat = strategy.operate(operation, new_model_config, workspace)
    return stat

def budget_run(budget_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    wede_acc_list, acquired_error_list, wede_train_error_list = [], [], []
    for budget in budget_list:
        operation.acquisition.budget = budget
        budget_stat = run(operation, new_model_config, workspace)
        wede_acc_list.append(budget_stat.wede_acc)
        acquired_error_list.append(budget_stat.acquired_error)
        wede_train_error_list.append(budget_stat.wede_train_error)
    return {
        'wede_acc': wede_acc_list,
        'acquired_error': acquired_error_list,
        'wede_train_error': wede_train_error_list
    }

def method_run(budget_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
    workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)

    budget_stat = budget_run(budget_list, operation, new_model_config, workspace)

    return budget_stat

def epoch_run(parse_args:ParseArgs, budget_list, dataset:dict, epo, operation: Config.Operation):
    old_model_config, new_model_config, general_config = Config.get_configs(epo, parse_args)

    workspace = WorkSpace(old_model_config, dataset, general_config)

    logger.info('Set up WorkSpace')
    
    workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)

    if parse_args.filter_market:
        known_labels = general_config['data']['labels']['cover']['target']
        workspace.set_market(operation.detection.vit, known_labels)
   
    stat = method_run(budget_list, new_model_config, operation, workspace)
    return stat

def main(epochs, device, detector_name, model_dir, utility_estimator, filter_market=False):
    fh = logging.FileHandler('log/{}/seq_stat.log'.format(model_dir),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    acquisition_method = 'seq'

    logger.info('Filter Market: {}, Estimator: {}'.format(filter_market, utility_estimator))

    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor, 'correctness')
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    budget_list = parse_args.general_config['data']['budget']

    wede_acc_list, acquired_error_list, wede_train_error_list = [], [], []

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = dataset_list[epo]
        stat = epoch_run(parse_args, budget_list, dataset, epo, operation)

        wede_acc_list.append(stat['wede_acc'])
        acquired_error_list.append(stat['acquired_error'])
        wede_train_error_list.append(stat['wede_train_error'])
    
    wede_acc_list = np.array(wede_acc_list)
    acquired_error_list = np.array(acquired_error_list)
    wede_train_error_list = np.array(wede_train_error_list)

    logger.info('wede acc')
    for idx, n_data in enumerate(budget_list):
        wede_acc = np.mean(wede_acc_list[:, idx, :], axis=0)
        logger.info('{}: [{}, {}],'.format(n_data, wede_acc[0], wede_acc[1]))
    
    logger.info('acquired error')
    for idx, n_data in enumerate(budget_list):
        acquired_error = np.mean(acquired_error_list[:, idx, :], axis=0)
        logger.info('{}: [{}, {}],'.format(n_data, acquired_error[0], acquired_error[1]))

    logger.info('wede train error')
    for idx, n_data in enumerate(budget_list):
        wede_train_error = np.mean(wede_train_error_list[:, idx, :], axis=0)
        logger.info('{}: [{}, {}],'.format(n_data, wede_train_error[0], wede_train_error[1]))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")
    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, utility_estimator=args.utility_estimator)