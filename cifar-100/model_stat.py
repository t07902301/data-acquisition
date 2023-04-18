from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter

def get_train_max_dv(train_loader, clf, clip_processor, n_cls):
    train_info, _ = CLF.apply_CLF(clf, train_loader, clip_processor)
    max_dv = []
    for c in range(n_cls):
        cls_indices = acquistion.extract_class_indices(c, train_info['gt'])
        train_cls_dv = train_info['dv'][cls_indices]
        max_dv.append(np.max(train_cls_dv))
    return max_dv

def threshold_run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, methods, new_img_num_list, product:Checker.subset, data_loader):
    result_list = []
    train_max_dv = get_train_max_dv(data_loader['train'], product.clf, product.clip_processor, model_config.class_number)
    for method in methods:
        print('In method', method)
        result_method = []
        for new_img_num in new_img_num_list:
            # acquisition_config.set_items('dv',new_img_num)
            acquisition_config.set_items(method,new_img_num)
            bound = Subset.get_threshold(product.clf, product.clip_processor, acquisition_config, model_config, data_loader['market'])
            # acquisition_config.set_items(method,new_img_num)
            bound = Subset.max_dv(bound, train_max_dv, model_config.pure)
            check_result = product.run(acquisition_config, bound)
            result_method.append(check_result)
        result_list.append(result_method)
    return result_list

def total_run(acquisition_config:Config.Acquistion, methods, new_img_num_list, product:Checker.total):
    result_list = []
    for method in methods:
        print('In method', method)
        result_method = []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            check_result = product.run(acquisition_config)
            result_method.append(check_result)
        result_list.append(result_method)
    return result_list

def run(check_method, new_img_num_list, acquire_instruction, methods, new_model_config, data_loader, product:Checker.prototype):
    if check_method == 'total':
        return total_run(acquire_instruction, methods, new_img_num_list, product)
    else:
        return threshold_run(acquire_instruction, new_model_config, methods, new_img_num_list, product, data_loader)

# def seq_threshold_run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, new_img_num_list, product:test_subset_checker):
#     result_list = []
#     for n_img in new_img_num_list:
#         acquisition_config.set_items('seq_clf',n_img)
#         model_config.set_path(acquisition_config)
#         threshold = seq_dv_bound(model_config, acquisition_config)
#         loader = threshold_test_subet_setter().get_subset_loders(product.test_info,threshold)
#         product.set_test_loader(loader)
#         check_result = product.run(acquisition_config)
#         result_list.append(check_result)
#     return result_list

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, pure, device)
    results = []

    method_list = ['dv','sm','conf','mix','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential with only SVM updates']

    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    if check_method == 'total' and pure == False:
        method_list = ['dv','sm','conf','mix','seq_clf', 'seq']
        method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential with only SVM updates', 'SVM and model updates']
        # method_list = ['dv','sm','conf','mix']
        # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    # threshold_collection = [-0.75,-0.5,-0.25,0]
    # threshold_collection = [-0.75,-0.5,-0.25,0, -0.5, -0.4, -0.3]
    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']
    # method_list, method_labels = ['seq_clf', 'seq'], ['seq_clf', 'seq']
    # new_img_num_list = [150,200]

    ds_list = Dataset.get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 

        product = Checker.factory(check_method, new_model_config)
        product.setup(old_model_config, ds)

        result_epoch = run(check_method, new_img_num_list, acquire_instruction, method_list, new_model_config, ds.loader, product)
        results.append(result_epoch)

    result_plotter = Plotter.ModelAcc(new_model_config)
    result_plotter.run(results, method_labels, new_img_num_list, check_method)  

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-m','--check_methods',type=str, default='threshold')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=Config.str2bool, default=False)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, check_method=args.check_methods,device=args.device, augment=args.augment)