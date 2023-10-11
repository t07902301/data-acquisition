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
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation) #probab / ensemble
        result_epoch, bound_stat = epoch_run(new_img_num_list, method_list, operation, checker)
        results.append(result_epoch)
        bound_stat_list.append(bound_stat)
    return results, bound_stat_list

def main(epochs, acquisition_method, device, detector_name, model_dir, stream_name, base_type, probab_bound):
    fh = logging.FileHandler('log/{}/test_{}.log'.format(model_dir, acquisition_method),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    new_model_setter = 'retrain'
    pure = True
    
    if acquisition_method == 'rs':
        probab_bound, stream_name = 0, 'probab'

    # if  dev_name == 'refine':
    #     acquisition_method, new_model_setter, pure, probab_bound, stream_name = 'dv', 'refine', False, 0, , 'probab'

    logger.info('Stream Name: {}, Probab Bound:{}'.format(stream_name, probab_bound))

    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name=stream_name)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition(seq_config=config['data']['seq']) if 'seq' in config['data'] else Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)

    result, bound_stat = bound_run(epochs, parse_args, ds_list, config['data']['n_new_data'], acquisition_method, operation)
    result = np.array(result)
    
    logger.info(acquisition_method)
    logger.info('avg:{}'.format(np.round(np.mean(result, axis=0), decimals=3).tolist()))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name) _ task _ (other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")
    parser.add_argument('-pd','--probab_bound',type=float,default=0.5, help='A bound of the probability from Cw to assign test set and create corresponding val set for model training.')
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv: u-wfs, rs: random, conf: confiden-score, seq: sequential u-wfs, pd: u-wfsd, seq_pd: sequential u-wfsd")
    parser.add_argument('-s','--stream',type=str, default='probab', help="Ensemble Method; dstr: AE, probab: WTA")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, acquisition_method=args.acquisition_method, stream_name=args.stream, base_type=args.base_type, probab_bound=args.probab_bound)
