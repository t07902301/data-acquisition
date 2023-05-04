from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.plot as Plotter

def test_shift_intervention(acquisition_config:Config.Acquistion, model_config:Config.NewModel, dataset_splits:Dataset.DataSplits, old_model_config:Config.OldModel):
    base_model = Model.load(old_model_config)
    clf = CLF.SVM(dataset_splits.loader['train_clip'])
    _ = clf.fit(base_model, dataset_splits.loader['val_shift'])
    bound = Subset.get_threshold(clf, acquisition_config, model_config, dataset_splits)
    print(bound)
    test_dv, _ = clf.predict(dataset_splits.loader['test_shift'])
    test_shift_mask = (test_dv<=bound)
    test_shift_fine_labels = Dataset.get_ds_labels(dataset_splits.dataset['test_shift'])[test_shift_mask]
    remove_fine_labels = config['data']['remove_fine_labels']   
    shift_cnt = 0 
    for label in remove_fine_labels:
        shift_cnt += (test_shift_fine_labels==label).sum()    
    return np.round(shift_cnt/test_shift_mask.sum(), decimals=3)*100

def shift_intervention(acquisition_config:Config.Acquistion, model_config:Config.NewModel, dataset_splits:Dataset.DataSplits):
    new_data = Log.get_log_data(acquisition_config, model_config, dataset_splits)
    remove_fine_labels = config['data']['remove_fine_labels']    
    shift_cnt = 0
    new_data_fine_labels = Dataset.get_ds_labels(new_data)
    for label in remove_fine_labels:
        shift_cnt += (new_data_fine_labels==label).sum()
    return np.round(shift_cnt/len(new_data), decimals=3)*100

def run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, methods, new_img_num_list, data_splits:Dataset.DataSplits, old_model_config:Config.OldModel):
    result_list, test_list = [], []
    for method in methods:
        print('In method', method)
        result_method, test_method = [], []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            score = shift_intervention(acquisition_config, model_config, data_splits)
            test_score = test_shift_intervention(acquisition_config, model_config, data_splits, old_model_config)
            result_method.append(score)
            test_method.append(test_score)
        result_list.append(result_method)
        test_list.append(test_method)
    return result_list, test_list


def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    results = []
    test_results = []

    method_list = ['dv','sm','conf','mix','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential with only SVM updates']
    # method_list, method_labels = ['seq_clf'], ['seq']
    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)

        result_epoch, test_epo = run(acquire_instruction, new_model_config, method_list, new_img_num_list, dataset_splits, old_model_config)
        results.append(result_epoch)
        test_results.append(test_epo)

    result_plotter = Plotter.Line(new_model_config)
    result_plotter.run(results, method_labels, new_img_num_list, 'shift proportion in new data','shift')  

    result_plotter = Plotter.Line(new_model_config)
    result_plotter.run(test_results, method_labels, new_img_num_list, 'shift proportion in test subset','test_shift')  

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