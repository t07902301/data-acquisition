from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter
import utils.objects.Config as Config 
import utils.acquistion as acquistion

def get_distribution(data_splits:Dataset.DataSplits, acquire_instruction:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, check_class, distribution_type):
    if distribution_type == 'total':
        distribution = total(check_class, data_splits, product.clf, product.clip_processor)
    else:
        distribution = threshold(acquire_instruction, model_config, product, data_splits.loader['market'], check_class)     
    return distribution      

def total(check_class, data_splits:Dataset.DataSplits, clf, clip_processor):
    split_dv = {}
    used_splits = ['train', 'val', 'market', 'test', 'train_clip']
    for split_name in used_splits:
        split_info, _ = CLF.apply_CLF(clf, data_splits.loader[split_name], clip_processor)
        cls_indices = acquistion.extract_class_indices(check_class, split_info['gt'])
        split_dv[split_name] = split_info['dv'][cls_indices]
    return split_dv
    
def threshold(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, market_loader, check_class):
    marker_info, _ = CLF.apply_CLF(product.clf, market_loader, product.clip_processor)
    train_distribution = subset_by_indicesLog(acquisition_config, model_config, marker_info, check_class)
    test_distribution = subset_by_threshold(acquisition_config, model_config, product, market_loader, check_class)
    dv_result = {
        'train': train_distribution,
        'test': test_distribution
    }
    return dv_result

def subset_by_indicesLog(acquisition_config:Config.Acquistion, model_config:Config.NewModel, market_info, check_class):
    idx_log_config = log.get_sub_log('indices', model_config, acquisition_config)
    idx_log_config.set_path(acquisition_config)
    new_data_indices = log.load(idx_log_config)        
    subset_dv = market_info['dv'][new_data_indices[check_class]] 
    return subset_dv  

def subset_by_threshold(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, market_loader, check_class):
    market_bound = Subset.get_threshold(product.clf, product.clip_processor, acquisition_config, model_config, market_loader)
    subset_loader = product.get_subset_loader(market_bound, acquisition_config)
    subset_info, _ = CLF.apply_CLF(product.clf, subset_loader['new_model'], product.clip_processor)
    subset_dv = subset_info['dv']
    cls_indices = acquistion.extract_class_indices(check_class, subset_info['gt'])
    return subset_dv[cls_indices]

def main(epochs, method, n_data, acquistion_class, new_model_setter='retrain', pure=False, model_dir ='', distribution_type='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, pure, device)
    results = []
    ds_list = Dataset.get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        acquire_instruction.set_items(method, n_data)
        product = Checker.factory('threshold', new_model_config)
        product.setup(old_model_config, ds)          
        distribution = get_distribution(ds, acquire_instruction, new_model_config, product, acquistion_class, distribution_type)
        results.append(distribution)
 
    plotter = Plotter.Distribution(new_model_config)
    plotter.run(epochs, results, distribution_type, acquistion_class, method, n_data)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=Config.str2bool, default=False)
    parser.add_argument('-dt','--distribution_type',type=str,default='total')
    parser.add_argument('-am','--acquistion_method',type=str, default='dv')
    parser.add_argument('-an','--acquistion_number',type=int, default=150)
    parser.add_argument('-ac','--acquistion_class',type=int, default=0)

    args = parser.parse_args()
    main(args.epochs, method=args.acquistion_method, n_data=args.acquistion_number, pure=args.pure,model_dir=args.model_dir, distribution_type=args.distribution_type,device=args.device, augment=args.augment,acquistion_class=args.acquistion_class)