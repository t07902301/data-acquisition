from utils.strategy import *
from utils.set_up import *
from utils.logging import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.stat_mode = True
    stat_results = strategy.operate(operation, new_model_config, workspace)
    return stat_results

def budget_run(new_img_num_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    budget_results = []
    for new_img_num in new_img_num_list:
        operation.acquisition.budget = new_img_num
        stat_results = run(operation, new_model_config, workspace)
        budget_results.append(stat_results)
    return budget_results

def method_run(method, new_img_num_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
    workspace.set_detector(operation.detection)
    workspace.set_validation(new_model_config.new_batch_size)

    operation.acquisition.method = method
    operation.acquisition = Config.AcquisitionFactory(operation.acquisition)

    budget_results = budget_run(new_img_num_list, operation, new_model_config, workspace)

    return budget_results

def epoch_run(parse_args, method, n_data_list, dataset:dict, epo, operation: Config.Operation):

    model_dir, device_config, base_type, pure, new_model_setter, config, filter_market = parse_args
    batch_size = config['hparams']['batch_size']
    superclass_num = config['hparams']['superclass']

    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
    workspace = WorkSpace(old_model_config, dataset, config)

    logger.info('Set up WorkSpace')
    
    workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)

    if filter_market:
        known_labels = config['data']['labels']['cover']['target']
        workspace.set_market(operation.detection.vit, known_labels)
   
    detect_acc = method_run(method, n_data_list, new_model_config, operation, workspace)
    return detect_acc

def bound_run(parse_args, epochs, ds_list, method_list, bound, budget_list, operation: Config.Operation):

    operation.acquisition.bound = bound
    detect_acc_list = []

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        detect_acc = epoch_run(parse_args, method_list, budget_list, dataset, epo, operation)
        detect_acc_list.append(detect_acc)
    detect_acc_list = np.array(detect_acc_list)

    for idx, n_data in enumerate(budget_list):
        avg_stat = np.mean(detect_acc_list[:, idx, :], axis=0)
        logger.info('{}: [{}, {}],'.format(n_data, avg_stat[0], avg_stat[1]))

def main(epochs, device, detector_name, model_dir, base_type, ensemble_criterion, ensemble_name, filter_market=False):

    fh = logging.FileHandler('log/{}/seq_stat.log'.format(model_dir),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    pure, new_model_setter = True, 'retrain'
    acquisition_method = 'seq'

    logger.info('Filter Market: {}'.format(filter_market))

    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(ensemble_criterion, ensemble_name, pdf='kde')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition(seq_config=config['data']['seq']) 
    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config, filter_market)
    bound_run(parse_args, epochs, ds_list, acquisition_method, None, config['data']['budget'], operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-ec','--criterion',type=float,default=0.5, help='Criterion in WTA ensemble')
    parser.add_argument('-em','--ensemble',type=str, default='ae', help="Ensemble Method")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, base_type=args.base_type, ensemble_criterion=args.ensemble_criterion,ensemble_name=args.ensemble)