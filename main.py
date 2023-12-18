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

def epoch_run(parse_args, budget_list, dataset:dict, epo, operation: Config.Operation):

    model_dir, device_config, base_type, pure, new_model_setter, config, filter_market = parse_args
    batch_size = config['hparams']['source']['batch_size']
    superclass_num = config['hparams']['source']['superclass']

    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
    workspace = WorkSpace(old_model_config, dataset, config)

    logger.info('Set up WorkSpace')
    
    workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)

    if filter_market:
        known_labels = config['data']['labels']['cover']['target']
        workspace.set_market(operation.detection.vit, known_labels)
   
    method_run(budget_list, new_model_config, operation, workspace)

def main(epochs, acquisition_method, device, detector_name, model_dir, base_type, utility_estimator, filter_market=False):

    fh = logging.FileHandler('log/{}/{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    pure, new_model_setter = True, 'retrain'

    logger.info('Filter Market: {}, Estimator: {}'.format(filter_market, utility_estimator))

    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config, filter_market)

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        epoch_run(parse_args, config['data']['budget'], dataset, epo, operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")

    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")

    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, base_type=args.base_type, utility_estimator=args.utility_estimator)