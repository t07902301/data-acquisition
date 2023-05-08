from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter

def threshold_run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, methods, new_img_num_list, product:Checker.subset, data_splits:Dataset.DataSplits):
    result_list = []
    for method in methods:
        print('In method', method)
        result_method = []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            bound = Subset.get_threshold(product.clf, acquisition_config, model_config, data_splits)
            check_result = product.run(acquisition_config, bound)
            result_method.append(check_result)
        result_list.append(result_method)
    return result_list

def run(new_img_num_list, acquire_instruction, methods, new_model_config, data_splits:Dataset.DataSplits, product:Checker.prototype):
    return threshold_run(acquire_instruction, new_model_config, methods, new_img_num_list, product, data_splits)

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    results = []

    method_list = ['dv','sm','conf','mix','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential with only SVM updates']
    # method_list, method_labels = ['dv', 'seq_clf'], ['dv', 'seq_clf']
    # method_list, method_labels = ['seq_clf'], ['seq_clf']

    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)

        product = Checker.factory(check_method, new_model_config)
        product.setup(old_model_config, dataset_splits)

        result_epoch = run(new_img_num_list, acquire_instruction, method_list, new_model_config, dataset_splits, product)
        results.append(result_epoch)

    result_plotter = Plotter.Line(new_model_config)
    result_plotter.run(results, method_labels, new_img_num_list, 'model accuracy change(%)', 'acc')  

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-m','--check_methods',type=str, default='ts')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=Config.str2bool, default=False)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, check_method=args.check_methods,device=args.device, augment=args.augment)