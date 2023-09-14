from utils.strategy import *
from utils.set_up import *
from utils.logging import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.operate(operation, new_model_config, workspace)

def data_run(new_img_num_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    for new_img_num in new_img_num_list:
        operation.acquisition.n_ndata = new_img_num
        run(operation, new_model_config, workspace)

def method_run(method, new_img_num_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):

    if method != 'rs':
        workspace.set_detector(operation.detection)
        workspace.set_validation(operation.stream, new_model_config.batch_size, new_model_config.new_batch_size)
        _, detect_acc= workspace.detector.predict(workspace.data_split.loader['test_shift'], workspace.base_model, True)

    else:
        workspace.validation_loader = workspace.data_split.loader['val_shift']
        logger.info('Keep val_shift for validation_loader') # Align with inference on the test set
        detect_acc = 0

    operation.acquisition.method = method
    operation.acquisition = Config.AcquisitionFactory(operation.acquisition)
    data_run(new_img_num_list, operation, new_model_config, workspace)

    return detect_acc

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

def bound_run(parse_args, epochs, ds_list, method_list, bound, n_new_data_list, operation: Config.Operation):

    operation.acquisition.bound = bound
    detect_acc_list = []

    for epo in range(epochs):
        logger.info('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        detect_acc = epoch_run(parse_args, method_list, n_new_data_list, dataset, epo, operation)
        detect_acc_list.append(detect_acc)

    logger.info('Detect Acc: {}'.format(np.mean(detect_acc_list)))
    logger.info('All: {}'.format(detect_acc_list))

def main(epochs, dev_name, device, detector_name, model_dir, base_type, option, dataset_name, filter_market=False):

    fh = logging.FileHandler('log/{}/{}.log'.format(model_dir, dev_name),mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    pure, new_model_setter = True, 'retrain'

    if dev_name == 'rs':
        method, probab_bound = dev_name, 0
    elif dev_name == 'refine':
        method, new_model_setter, pure, probab_bound = 'dv', 'refine', False, 0
    else:
        method, probab_bound = dev_name, 0.5 

    logger.info('Filter Market: {}'.format(filter_market))

    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device, option, dataset_name)
    
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition(seq_config=config['data']['seq']) if 'seq' in config['data'] else Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)

    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config, filter_market)
    bound_run(parse_args, epochs, ds_list, method, None, config['data']['n_new_data'], operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-op','--option',type=str, default='object')
    parser.add_argument('-ds','--dataset',type=str, default='core')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, dev_name=args.dev, base_type=args.base_type, option=args.option, dataset_name=args.dataset)