from utils.strategy import *
from utils.set_up import *
from utils.logging import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.operate(operation, new_model_config, workspace)

def budget_run(budget_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)
    for budget in budget_list:
        operation.acquisition.set_budget(budget)
        run(operation, new_model_config, workspace)

def method_run(budget_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
    
    # method = operation.acquisition.method
    # if method != 'rs':
    #     workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)
    #     workspace.set_validation(new_model_config.new_batch_size)
    # else:
    #     workspace.validation_loader = workspace.data_split.loader['val_shift']
    #     logger.info('Keep val_shift for validation_loader') # Align with inference on the test set

    budget_run(budget_list, operation, new_model_config, workspace)

def epoch_run(parse_args:ParseArgs, budget_list, dataset:dict, epo, operation: Config.Operation):

    old_model_config, new_model_config, general_config = Config.get_configs(epo, parse_args)
    
    workspace = WorkSpace(old_model_config, dataset, general_config)

    logger.info('Set up WorkSpace')

    workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)

    if parse_args.filter_market:
        known_labels = general_config['data']['labels']['cover']['target']
        workspace.set_market(operation.detection.vit, known_labels)
   
    method_run(budget_list, new_model_config, operation, workspace)

def main(epochs, acquisition_method, device, detector_name, model_dir, utility_estimator, filter_market=False):

    fh = logging.FileHandler('log/{}/{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Filter Market: {}, Estimator: {}'.format(filter_market, utility_estimator))

    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)
    parse_args.filter_market = filter_market
    
    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=parse_args.general_config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = dataset_list[epo]
        epoch_run(parse_args, parse_args.general_config['data']['budget'], dataset, epo, operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")

    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")

    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, utility_estimator=args.utility_estimator)