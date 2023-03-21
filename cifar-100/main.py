from utils.strategy import *

def run(ds:DataSplits, old_model_config, new_model_config, acquisition_config:AcquistionConfig, methods, new_img_num_list):
    subset_length_record = {
        k: len(ds.dataset[k]) for k in ds.dataset.keys() 
    }

    percent_epoch = []

    for method in methods:
        print('In method', method)
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            if method not in ['seq','seq_clf']:
                strategy = NonSeqStrategy(ds,old_model_config, new_model_config)
                new_model = strategy.get_new_model(acquisition_config)
                # new_model = non_seq(ds, old_model_config, new_model_config, acquisition_config)
            else:
                strategy = SeqStrategy(ds,old_model_config, new_model_config)
                new_model, minority_cnt = strategy.get_new_model(acquisition_config)
                # new_model, minority_cnt = seq(ds, old_model_config, new_model_config, seq_acq_config)
                percent_epoch.append(minority_cnt/new_img_num)

            new_model_config.set_path(acquisition_config)
            save_model(new_model,new_model_config.path)

            del new_model

            for split in ds.dataset.keys():
                assert subset_length_record[split] == len(ds.dataset[split]), 'init {} is changed'.format(split)

    return percent_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', methods='', seq_rounds=1):
    print('Use pure: ',pure)
    percent_list = []
    method_list = ['dv','sm','conf','mix'] if methods=='non_seq' else [methods]
    # method_list =['conf']
    # new_img_num_list = [50]
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, pure)

    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = DataSplits(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            ds.modify_coarse_label(label_map)

        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir,pure, new_model_setter)
        acquistion_config = AcquistionConfigFactory(methods,epo, seq_rounds)

        percent_epoch = run(ds,old_model_config,new_model_config, acquistion_config, method_list, new_img_num_list)
        percent_list.append(percent_epoch)

    percent_list = np.array(percent_list)
    print(np.round(np.mean(percent_list,axis=0),decimals=3))
    print(data_config)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')
    parser.add_argument('-r','--rounds',type=int,default=2)
    parser.add_argument('-m','--methods',type=str)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,seq_rounds=args.rounds,methods=args.methods)
    # print(args.method)