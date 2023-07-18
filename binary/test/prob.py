import sys
sys.path.append('/home/yiwei/data-acquisition/binary/')
# sys.path.append('..')
from utils.strategy import *
import utils.statistics.checker as Checker
from utils.set_up import set_up

def method_run(n_img_list, operation:Config.Operation, checker: Checker.prototype):
    acc_change_list = []
    for n_img in n_img_list:
        operation.acquisition.n_ndata = n_img
        acc_change = n_data_run(operation, checker)
        acc_change_list.append(acc_change)
    return acc_change_list

def n_data_run(operation:Config.Operation, checker: Checker.prototype):
    check_result = checker.run(operation)
    return check_result

def run(operation:Config.Operation, methods, new_img_num_list, checker:Checker.prototype):
    result_list = []
    for method in methods:
        print('In method', method)
        operation.acquisition.method = method
        operation.acquisition = Config.AcquisitionFactory(method, operation.acquisition)
        result_method = method_run(new_img_num_list, operation, checker)
        result_list.append(result_method)
    return result_list

# def check_bound(old_model_config: Config.OldModel, datasplit: Dataset.DataSplits, acquire_instruction: Config.Acquisition, clip_processor):
#     base_model = Model.resnet(2)
#     base_model.load(old_model_config)
#     clf = Detector.SVM(datasplit.loader['train_clip'], clip_processor)
#     _ = clf.fit(base_model, datasplit.loader['val_shift'])
#     market_dv, _ = clf.predict(datasplit.loader['market'])
#     return (market_dv <= acquire_instruction.bound).sum()

def epoch_run(new_img_num_list, method_list, operation: Config.Operation, checker: Checker.prototype):
    result_epoch = run(operation, method_list, new_img_num_list, checker)
    # bound_stat = check_bound(old_model_config, dataset_splits,acquire_instruction, clip_processor)
    return result_epoch, 0

def bound_run(epochs, parse_args, dataset_list, new_img_num_list, method_list, operation: Config.Operation):
    results = []
    bound_stat_list = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, plot=False) #probab / ensemble
        result_epoch, bound_stat = epoch_run(new_img_num_list, method_list, operation, checker)
        results.append(result_epoch)
        bound_stat_list.append(bound_stat)
    return results, bound_stat_list

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = ''):
    print('Detector:', detector_name)
    if pure == False:
        probab_bound = 0
        stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    else:
        stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='ensemble')
    print('Probab bound:', probab_bound)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device)
    method_list = ['dv','sm','conf', 'seq_clf'] if new_model_setter!='refine' else ['dv']
    # method_list = ['dv','sm','conf', 'seq_clf'] 
    # method_list = ['dv']

    clip_processor = Detector.load_clip(device_config)
    parse_args = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)

    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    # bounds = [-1.5, -1, -0.8, -0.5, 0, 0.5]
    bounds = [None]
    for bound in bounds:
        acquire_instruction.bound = bound
        result, bound_stat = bound_run(epochs, parse_args, ds_list, new_img_num_list, method_list, operation)
        result = np.array(result)
        print(result.shape)
        for idx, method in enumerate(method_list):
            method_result = result[:, idx, :]
            print(method)
            print(*np.round(np.mean(method_result, axis=0), decimals=3).tolist(), sep=',')

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-ns','--new_model_setter',type=str,default='retrain')
    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name, new_model_setter=args.new_model_setter)