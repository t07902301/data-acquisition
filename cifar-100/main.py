from utils.strategy import *

def run(ds:DataSplits, methods_list, new_img_num_list, old_model_config:OldModelConfig, new_model_config:NewModelConfig, acquire_instruction:AcquistionConfig):
    subset_length_record = {
        k: len(ds.dataset[k]) for k in ds.dataset.keys() 
    }

    percent_epoch = []
    for method in methods_list:
        strategy = StrategyFactory(method)
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items(method,new_img_num)
            strategy.operate(acquire_instruction, ds, old_model_config, new_model_config)
            for split in ds.dataset.keys():
                assert subset_length_record[split] == len(ds.dataset[split]), 'init {} is changed'.format(split)

    return percent_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', strategy='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)

    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = parse_config(model_dir, pure)
    print_config(batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    percent_list = []

    method_list = ['dv','sm','conf','mix'] if strategy=='non_seq' else [strategy]
    # method_list = ['seq']
    # method_list, new_img_num_list = ['dv','sm'], [150,200]
    # new_img_num_list = [150,200,]

    ds_list = get_data_splits_list(epochs, select_fine_labels, label_map, ratio)

    for epo in range(epochs):
        print('In epoch {}'.format(epo))
        ds = ds_list[epo]
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter, augment)
        acquire_instruction = AcquistionConfigFactory(strategy,seq_rounds_config)
        percent_epoch = run(ds,method_list, new_img_num_list, old_model_config,new_model_config, acquire_instruction)
        percent_list.append(percent_epoch)

    percent_list = np.array(percent_list)
    print(np.round(np.mean(percent_list,axis=0),decimals=3))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-s','--strategy',type=str)
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=str2bool, default=True)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,strategy=args.strategy,device=args.device, augment=args.augment)
    # print(args.method)