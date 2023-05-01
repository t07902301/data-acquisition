from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter
import utils.objects.Config as Config 
import utils.acquistion as acquistion
from utils import n_workers

def get_distribution(dataset_splits:Dataset.DataSplits, acquire_instruction:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, check_class, distribution_type):
    if distribution_type == 'total':
        distribution = total(check_class, dataset_splits, product.clf)
    else:
        # distribution = threshold(acquire_instruction, model_config, product, data_splits.loader['market'], check_class)   
        distribution = compare(acquire_instruction, model_config, product, dataset_splits, check_class)  
    return distribution      

def total(check_class, data_splits:Dataset.DataSplits, clf):
    split_dv = {}
    used_splits = list(data_splits.dataset.keys())
    for split_name in used_splits:
        split_info, _ = clf.predict(data_splits.loader[split_name])
        cls_indices = acquistion.extract_class_indices(check_class, split_info['gt'])
        split_dv[split_name] = split_info['dv'][cls_indices]
    return split_dv
    
# def threshold(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, market_loader, check_class):
#     market_info, _ = CLF.apply_CLF(product.clf, market_loader, product.clip_processor)
#     if acquisition_config.method == 'seq_clf':
#         train_distribution = data_distribution(acquisition_config, model_config, check_class, product)
#     else:
#         train_distribution = indices_distribution(acquisition_config, model_config, market_info, check_class)
#     test_distribution = subset_distribution(acquisition_config, model_config, product, market_loader, check_class)
#     dv_result = {
#         'train': train_distribution,
#         'test': test_distribution
#     }
#     return dv_result

def compare(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, dataset_splits:Dataset.DataSplits, check_class):
    test_info, _ = product.clf.predict(dataset_splits.loader['test_shift'])
    test_distribution = test_info['dv'][test_info['gt']==check_class]
    init_train_info, _ = product.clf.predict(dataset_splits.loader['train_clip'])
    init_train_distribution = init_train_info['dv'][init_train_info['gt']==check_class]
    dv_result = {
        'old train': init_train_distribution,
        'test_shift': test_distribution
    }

    method_list = ['dv','sm','conf','mix', 'seq_clf', 'seq']
    for method in method_list:
        acquisition_config.method = method
        train_distribution = get_log_distribution(acquisition_config, model_config, product, dataset_splits, check_class)
        dv_result[method] = train_distribution
        print(np.max(train_distribution), np.min(train_distribution))

    return dv_result

def get_log_distribution(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, dataset_splits:Dataset.DataSplits, check_class):
    if 'seq' in acquisition_config.method:
        new_data = get_log_data(acquisition_config, model_config)
    else:
        new_data_indices = get_log_indices(acquisition_config, model_config)   
        new_data = torch.utils.data.Subset(dataset_splits.dataset['market'], new_data_indices)
    train_data = torch.utils.data.ConcatDataset([dataset_splits.dataset['train'],new_data])
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=model_config.batch_size, 
                            num_workers=n_workers)
    train_info, _ = product.clf.predict(train_data_loader)
    return train_info['dv'][train_info['gt']==check_class]

def get_log_data(acquisition_config:Config.Acquistion, model_config:Config.NewModel):
    log_config = Log.get_config(model_config, acquisition_config, 'data')
    log_config.set_path(acquisition_config)
    new_data = Log.load(log_config)  
    return new_data

def get_log_indices(acquisition_config:Config.Acquistion, model_config:Config.NewModel):
    log_config = Log.get_config(model_config, acquisition_config, 'indices')
    log_config.set_path(acquisition_config)
    new_data_indices = Log.load(log_config)  
    return new_data_indices   

def subset_distribution(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, market_loader, check_class):
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
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        acquire_instruction.set_items(method, n_data)
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)

        product = Checker.factory('threshold', new_model_config)
        product.setup(old_model_config, dataset_splits)   
           
        distribution = get_distribution(dataset_splits, acquire_instruction, new_model_config, product, acquistion_class, distribution_type)
        results.append(distribution)
 
    plotter = Plotter.Distribution(new_model_config)
    plotter.run(epochs, results, distribution_type, acquistion_class, method, n_data)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=False)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=Config.str2bool, default=False)
    parser.add_argument('-dt','--distribution_type',type=str,default='total')
    parser.add_argument('-am','--acquistion_method',type=str, default='dv')
    parser.add_argument('-an','--acquistion_number',type=int, default=150)
    parser.add_argument('-ac','--acquistion_class',type=int, default=0)

    args = parser.parse_args()
    main(args.epochs, method=args.acquistion_method, n_data=args.acquistion_number, pure=args.pure,model_dir=args.model_dir, distribution_type=args.distribution_type,device=args.device, augment=args.augment,acquistion_class=args.acquistion_class)