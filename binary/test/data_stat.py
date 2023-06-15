from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.checker as Checker
import utils.statistics.subset as Subset

def check_data(acquisition_config:Config.Acquistion, dataset_splits:Dataset.DataSplits, checker:Checker.prototype):
    model_config = checker.model_config
    model_config.set_path(acquisition_config)
    model = Model.resnet(2)
    model.load(model_config)
    new_data = Log.get_log_data(acquisition_config, model_config, dataset_splits)
    shift_score = Subset.new_cls_stat(new_data)
    new_data_loader = torch.utils.data.DataLoader(new_data, batch_size= model_config.new_batch_size)
    dv, _ = checker.clf.predict(new_data_loader)   
    return shift_score, np.max(dv)

def method_run(n_img_list, acquisition_config:Config.Acquistion, checker: Checker.prototype, dataset_splits:Dataset.DataSplits):
    acc_change_list = []
    for n_img in n_img_list:
        acquisition_config.n_ndata = n_img
        acc_change = n_data_run(acquisition_config, checker, dataset_splits)
        acc_change_list.append(acc_change)
    return acc_change_list

def n_data_run(acquisition_config, checker: Checker.prototype, dataset_splits:Dataset.DataSplits):
    check_result = check_data(acquisition_config, dataset_splits, checker)
    return check_result

def run(acquisition_config:Config.Acquistion, methods, new_img_num_list, checker:Checker.prototype, dataset_splits:Dataset.DataSplits):
    result_list = []
    for method in methods:
        print('In method', method)
        acquisition_config.method = method
        if 'seq' in method:
            checker.clf = Log.get_log_clf(acquisition_config, checker.model_config, checker.clip_set_up_loader, checker.clip_processor)
        result_method = method_run(new_img_num_list, acquisition_config, checker, dataset_splits)
        result_list.append(result_method)
    return result_list

def epoch_run(epoch, parse_para, dataset, clip_processor, new_img_num_list, method_list, acquire_instruction:Config.Acquistion):
    batch_size, superclass_num, model_dir, device_config, pure, new_model_setter, seq_rounds_config = parse_para
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, base_type='resnet')
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, pure, new_model_setter, batch_size['new'], base_type='resnet')
    dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)
    checker = Checker.factory('prob', new_model_config, clip_processor, dataset_splits.loader['train_clip'])
    checker.setup(old_model_config, dataset_splits)
    acquire_instruction.bound = 0
    result_epoch = run(acquire_instruction, method_list, new_img_num_list, checker, dataset_splits)
    return result_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0):
    print('Use pure: ',pure)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)

    # method_list, method_labels = ['dv','sm','conf','seq_clf'], ['greedy decision value','random sampling','model confidence','sequential with only SVM updates']
    method_list, method_labels = ['dv'], ['dv']
    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    clip_processor = Detector.load_clip(device_config)
    parse_para = (batch_size, superclass_num, model_dir, device_config, pure, new_model_setter, seq_rounds_config)
    acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
    results = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        result_epoch = epoch_run(epo, parse_para, ds_list[epo], clip_processor, new_img_num_list, method_list, acquire_instruction)
        results.append(result_epoch)
    
    print(results)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device)