from utils.strategy import *
from utils.set_up import *
from utils.logging import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace, stat_mode):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.seq_stat_mode = stat_mode
    stat = strategy.operate(operation, new_model_config, workspace)
    return stat

def budget_run(budget_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace, stat_mode):
    budget_stat = []
    for budget in budget_list:
        operation.acquisition.budget = budget
        stat = run(operation, new_model_config, workspace, stat_mode)
        budget_stat.append(stat)
    return budget_stat

def method_run(budget_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace, stat_mode):
    workspace.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf)
    workspace.set_validation(new_model_config.new_batch_size)

    budget_stat = budget_run(budget_list, operation, new_model_config, workspace, stat_mode)

    return budget_stat

def epoch_run(parse_args, budget_list, dataset:dict, epo, operation: Config.Operation, stat_mode):

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
   
    stat = method_run(budget_list, new_model_config, operation, workspace, stat_mode)
    return stat

def main(epochs, acquisition_method, device, detector_name, model_dir, base_type, utility_estimator, stat_mode, filter_market=False):
    fh = logging.FileHandler('log/{}/seq_stat.log'.format(model_dir),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    pure, new_model_setter = True, 'retrain'
    acquisition_method = 'seq'

    logger.info('Filter Market: {}, Estimator: {}'.format(filter_market, utility_estimator))

    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)
    
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble()
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'], utility_estimator=utility_estimator)

    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)

    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config, filter_market)

    stat_list = []
    budget_list = config['data']['budget']

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        stat = epoch_run(parse_args, budget_list, dataset, epo, operation, stat_mode)
        stat_list.append(stat)
    stat_list = np.array(stat_list)

    for idx, n_data in enumerate(budget_list):
        avg_stat = np.mean(stat_list[:, idx, :], axis=0)
        logger.info('{}: [{}, {}],'.format(n_data, avg_stat[0], avg_stat[1]))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: confiden-score, seq: sequential")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-m','--mode',type=str,default='wede_acc', help="plug-in stat for sequential acquisition")
    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, base_type=args.base_type, utility_estimator=args.utility_estimator, stat_mode=args.mode)