from utils.strategy import *

def run(ds:DataSplits, methods_list, new_img_num_list, old_model_config:OldModelConfig, new_model_config:NewModelConfig, acquire_instruction:AcquistionConfig):
    subset_length_record = {
        k: len(ds.dataset[k]) for k in ds.dataset.keys() 
    }

    percent_epoch = []
    for method in methods_list:
        strategy = StrategyFactory(old_model_config, new_model_config, method)
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items(method,new_img_num)
            strategy.operate(acquire_instruction, ds)
            for split in ds.dataset.keys():
                assert subset_length_record[split] == len(ds.dataset[split]), 'init {} is changed'.format(split)

    return percent_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', strategy='', seq_rounds=1):
    print('Use pure: ',pure)
    percent_list = []
    # method_list =['conf']
    # new_img_num_list = [50]
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, pure)
    method_list = ['dv','sm','conf','mix'] if strategy=='non_seq' else [strategy]
    # method_list = ['sm','mix']
    # method_list, new_img_num_list = ['dv','sm','conf'], [25,50,75]
    # new_img_num_list,method_list = [150], ['dv']

    ds_list = get_data_splits_list(epochs, select_fine_labels, model_dir, label_map)

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, epo, pure, new_model_setter)
        acquire_instruction = AcquistionConfigFactory(strategy,seq_rounds)
        percent_epoch = run(ds,method_list, new_img_num_list, old_model_config,new_model_config, acquire_instruction)
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
    parser.add_argument('-s','--strategy',type=str)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,seq_rounds=args.rounds,strategy=args.strategy)
    # print(args.method)