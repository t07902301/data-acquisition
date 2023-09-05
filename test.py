from utils.strategy import *
import utils.statistics.checker as Checker
from utils.set_up import *

def method_run(n_data_list, operation:Config.Operation, checker: Checker.Prototype):
    acc_change_list = []
    for n_data in n_data_list:
        operation.acquisition.n_ndata = n_data
        acc_change = n_data_run(operation, checker)
        acc_change_list.append(acc_change)
    return acc_change_list

def n_data_run(operation:Config.Operation, checker: Checker.Prototype):
    check_result = checker.run(operation)
    return check_result

def run(operation:Config.Operation, method, new_img_num_list, checker:Checker.Prototype):
    logger.info('In method: {}'.format(method))
    operation.acquisition.method = method
    operation.acquisition = Config.AcquisitionFactory(operation.acquisition)
    result = method_run(new_img_num_list, operation, checker)
    return result

# def check_bound(old_model_config: Config.OldModel, datasplit: Dataset.DataSplits, acquire_instruction: Config.Acquisition, clip_processor):
#     base_model = Model.resnet(2)
#     base_model.load(old_model_config)
#     clf = Detector.SVM(datasplit.loader['train_clip'], clip_processor)
#     _ = clf.fit(base_model, datasplit.loader['val_shift'])
#     market_dv, _ = clf.predict(datasplit.loader['market'])
#     return (market_dv <= acquire_instruction.bound).sum()

def epoch_run(new_img_num_list, method_list, operation: Config.Operation, checker: Checker.Prototype):
    result_epoch = run(operation, method_list, new_img_num_list, checker)
    # bound_stat = check_bound(old_model_config, dataset_splits,acquire_instruction, clip_processor)
    return result_epoch, 0

def bound_run(epochs, parse_args, dataset_list, new_img_num_list, method_list, operation: Config.Operation):
    results = []
    bound_stat_list = []
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, plot=False) #probab / ensemble
        result_epoch, bound_stat = epoch_run(new_img_num_list, method_list, operation, checker)
        results.append(result_epoch)
        bound_stat_list.append(bound_stat)
    return results, bound_stat_list

def main(epochs, dev_name, device, detector_name, model_dir, stream_name, base_type, option, dataset_name):
    fh = logging.FileHandler('log/{}/test_{}.log'.format(model_dir, dev_name),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    new_model_setter = 'retrain'
    pure = True
    
    if dev_name == 'rs':
        method, probab_bound, stream_name = dev_name, 0, 'probab'
        # method, probab_bound = dev_name, 0.5

    elif dev_name == 'refine':
        method, new_model_setter, pure, probab_bound = 'dv', 'refine', False, 0
    else:
        method, probab_bound = dev_name, 0.5

    logger.info('Stream Name: {}, Probab Bound:{}'.format(stream_name, probab_bound))

    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device, option, dataset_name)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name=stream_name)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition(seq_config=config['data']['seq']) if 'seq' in config['data'] else Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)

    result, bound_stat = bound_run(epochs, parse_args, ds_list, config['data']['n_new_data'], method, operation)
    result = np.array(result)
    
    logger.info(method)
    logger.info('avg:{}'.format(np.round(np.mean(result, axis=0), decimals=3).tolist()))
    
    logger.info(method)
    logger.info('all: {}'.format(result.tolist()))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-s','--stream',type=str, default='probab')
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-op','--option',type=str, default='object')
    parser.add_argument('-ds','--dataset',type=str, default='core')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, dev_name=args.dev, stream_name=args.stream, base_type=args.base_type, option= args.option, dataset_name=args.dataset)
