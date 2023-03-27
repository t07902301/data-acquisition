from utils.strategy import *
from utils.statistics import *

def run(acquisition_config:AcquistionConfig, methods, new_img_num_list, product:checker):
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

def threshold_run(acquisition_config:AcquistionConfig, methods, new_img_num_list, product:test_subset_checker):
    result_list = []
    for n_img in new_img_num_list:
        result_img = []
        threshold = product.threshold_collection[n_img]
        loader = threshold_test_subet_setter().get_subset_loders(product.test_info,threshold)
        product.set_test_loader(loader)
        for method in methods:
            acquisition_config.set_items(method,n_img)
            check_result = product.run(acquisition_config)
            result_img.append(check_result)
        result_list.append(result_img)
    return result_list

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='threshold'):
    print('Use pure: ',pure)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, pure)

    results = []
    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential', 'sequential with only SVM updates']
    # method_labels, method_list, new_img_num_list = ['dv', 'sm', 'conf'], ['dv', 'sm', 'conf'], [25,50,75,100]
    # new_img_num_list,method_list = [150, 200], ['dv','sm']
    # threshold_collection = [-0.75,-0.5,-0.25,0]
    # threshold_collection = [-0.75,-0.5,-0.25,0, -0.5, -0.4, -0.3]

    ds_list = get_data_splits_list(epochs, select_fine_labels, model_dir, label_map)
    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, epo, pure, new_model_setter)
        acquistion_config = AcquistionConfig()

        threshold_collection = get_threshold_collection(new_img_num_list, acquistion_config, new_model_config, old_model_config, ds, epo)
        # threshold_collection = None
        product = checker_factory(check_method, new_model_config, threshold_collection)
        product.setup(old_model_config, ds)

        if check_method =='threshold':
            result_epoch = threshold_run(acquistion_config, method_list, new_img_num_list, product)
        else:
            result_epoch = run(acquistion_config, method_list, new_img_num_list, product)

        results.append(result_epoch)
        # print(threshold_collection)

    result_plotter = plotter(check_method,method_labels,new_img_num_list, new_model_config)
    result_plotter.plot_data(results, threshold_collection)    
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')
    parser.add_argument('-cm','--check_methods',type=str)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, check_method=args.check_methods)