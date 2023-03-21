from utils.strategy import *
from utils.statistics import *

def run(acquisition_config:AcquistionConfig, methods, new_img_num_list, product:checker_factory):

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

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', check_method='threshold'):
    print('Use pure: ',pure)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, pure)

    results = []
    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential', 'sequential with only SVM updates']
    # method_list, new_img_num_list = ['dv'], [50,100]
    # method_labels = ['dv']
    # threshold_list = [-0.75,-0.5,-0.25,0]
    threshold_list = [-0.5, -0.4, -0.3]

    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = DataSplits(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            ds.modify_coarse_label(label_map)
        
        ds.get_dataloader(batch_size)

        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir,pure, new_model_setter)
        acquistion_config = AcquistionConfig(model_cnt=epo)

        product = checker_factory(check_method, new_model_config, threshold_list)
        product.setup(old_model_config, ds)

        result_epoch = run(acquistion_config, method_list, new_img_num_list, product)
        results.append(result_epoch)

    results = np.array(results)
    result_plotter = plotter(check_method,method_labels,new_img_num_list, new_model_config)
    result_plotter.plot_data(results, threshold_list)    
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')
    parser.add_argument('-m','--check_methods',type=str)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, check_method=args.check_methods)