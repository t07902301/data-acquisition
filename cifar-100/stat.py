from utils.strategy import *
from utils.statistics import *
from utils.set_up import set_up

def get_clf_statistics(methods_list, new_img_num_list, new_model_config:Config.NewModel, acquire_instruction:Config.Acquistion):
    shift_score_list = []
    clf_precision_list = []
    for method in methods_list:
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items(method,new_img_num)
            stat_config = log.get_sub_log('stat', new_model_config, acquire_instruction)
            clf_stat = log.load(stat_config)
            shift_score_list.append(clf_stat['cv score'])
            clf_precision_list.append(clf_stat['precision'])
    return shift_score_list, clf_precision_list

def threshold_run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, methods, new_img_num_list, product:test_subset_checker, market_ds):
    result_list = []
    for method in methods:
        print('In method', method)
        result_method = []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            bound = get_threshold(product.clf, product.clip_processor, acquisition_config, model_config, market_ds)
            check_result = product.run(acquisition_config, bound)
            result_method.append(check_result)
            # print(bound)
        result_list.append(result_method)
    return result_list

def total_run(acquisition_config:Config.Acquistion, methods, new_img_num_list, product:checker):
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

def run(check_method, new_img_num_list, acquire_instruction, methods, new_model_config, market_ds, product):
    if check_method == 'total':
        return total_run(acquire_instruction, methods, new_img_num_list, product)
    else:
        return threshold_run(acquire_instruction, new_model_config, methods, new_img_num_list, product, market_ds)

# def threshold_run(acquisition_config:Config.Acquistion, methods, new_img_num_list, product:test_subset_checker, threshold_collection):
#     result_list = []
#     for n_img in new_img_num_list:
#         result_img = []
#         threshold = threshold_collection[n_img]
#         loader = threshold_test_subet_setter().get_subset_loders(product.test_info,threshold)
#         product.set_test_loader(loader)
#         for method in methods:
#             acquisition_config.set_items(method,n_img)
#             check_result = product.run(acquisition_config)
#             result_img.append(check_result)
#         result_list.append(result_img)
#     return result_list

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

# def run(check_method, new_img_num_list, acquire_instruction, new_model_config, old_model_config, ds, product):
#     if check_method =='threshold':
#         method_list = ['dv','sm','conf','mix']
#         threshold_collection = get_threshold_collection(new_img_num_list, acquire_instruction, new_model_config, old_model_config, ds)
#         # threshold_collection = None        
#         result_epoch = threshold_run(acquire_instruction, method_list, new_img_num_list, product, threshold_collection)

#         seq_result_epoch = seq_threshold_run(acquire_instruction, new_model_config, new_img_num_list, product)
#         for idx in enumerate(new_img_num_list):
#             result_epoch[idx].append(seq_result_epoch[idx])
#     # else:
#     #     result_epoch = run(acquire_instruction, method_list, new_img_num_list, product)
#     return result_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, pure, device)
    results = []
    score, precision = [], []

    method_list = ['dv','sm','conf','mix','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential with only SVM updates']
    # method_labels, method_list, new_img_num_list = ['dv', 'sm', 'conf'], ['dv', 'sm', 'conf'], [25,50,75,100]
    # new_img_num_list,method_list, method_labels = [150, 200], ['dv','sm'], ['dv','sm']
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

        # product = checker_factory(check_method, new_model_config)
        # product.setup(old_model_config, ds)

        # result_epoch = run(check_method, new_img_num_list, acquire_instruction, method_list, new_model_config, ds.dataset['market'], product)
        # results.append(result_epoch)

        score_epo, prec_epo = get_clf_statistics(method_list, new_img_num_list, new_model_config, acquire_instruction)
        score.append(score_epo)
        precision.append(prec_epo)

    # result_plotter = plotter(check_method,method_labels,new_img_num_list, new_model_config)
    # result_plotter.plot_data(results)  
    # (batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio)

    CLF.statistics(epochs, score, precision)

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