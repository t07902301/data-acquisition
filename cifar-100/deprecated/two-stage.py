from utils.strategy import *
import matplotlib.pyplot as plt

def run(ds:DataSplits, old_model_config:OldModelConfig, new_model_config:NewModelConfig, acquisition_config:AcquistionConfig, method_list, new_img_num_list, ds_setter:test_subet_setter):
    ds.get_dataloader(new_model_config.batch_size)
    # Load base model
    base_model = load_model(old_model_config.path)
    # Get SVM 
    clf,clip_processor,_ = get_CLF(base_model,ds.loader)

    test_info = apply_CLF(clf,base_model,ds.loader['test'],clip_processor)
    base_acc = (test_info['gt']==test_info['pred']).mean()*100

    test_info['batch_size'] = old_model_config.batch_size
    test_info['dataset'] = ds.dataset['test']

    ds_setter.get_dataset_info(test_info)
    test_loaders = ds_setter.get_subset()
    acc_change = []

    # init_test_loader = torch.utils.data.DataLoader(ds.dataset['test'], batch_size=test_info['batch_size'], num_workers=config['num_workers'])

    for method in method_list:
        acc_method = []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            new_model_config.set_path(acquisition_config)
            new_model = load_model(new_model_config.path)
            acc_img = []
            for loader in test_loaders:

                gt,pred,_  = evaluate_model(loader['new_model'],new_model)
                new_correct = (gt==pred)

                gt,pred,_  = evaluate_model(loader['old_model'],base_model)
                old_correct = (gt==pred)

                total_correct = np.concatenate((old_correct,new_correct))
                assert total_correct.size == test_info['gt'].size
                acc_img.append(total_correct.mean()*100-base_acc)
            acc_method.append(acc_img)
        acc_change.append(acc_method)
    return acc_change

def main(epochs,new_model_setter='retrain', pure=False, model_dir ='', subset_method='threshold'):
    print('Use pure: ',pure)
    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential', 'sequential with only SVM updates']
    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    # method_list = ['sm']
    threshold_list = [-0.75,-0.5,-0.25,0]
    # threshold_list = [-0.75]

    acc_list = []
    # img_per_cls_list = [50]

    batch_size, select_fine_labels, label_map, img_per_cls_list, superclass_num = parse_config(model_dir, pure)

    for model_cnt in range(epochs):
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, model_cnt)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, pure, new_model_setter)
        acquistion_config = AcquistionConfig(model_cnt=model_cnt)

        print('epoch',model_cnt)
        ds = DataSplits(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            ds.modify_coarse_label(label_map)

        test_subset = test_subet_setter(subset_method,threshold_list)

        acc_epoch = run(ds, old_model_config, new_model_config, acquistion_config, method_list, img_per_cls_list, test_subset)

        acc_list.append(acc_epoch)



    acc_list = np.round(acc_list,decimals=3)

    result_plotter = plotter(subset_method,method_labels,img_per_cls_list)
    result_plotter.plot_data(acc_list,new_model_config,threshold_list)


import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir)