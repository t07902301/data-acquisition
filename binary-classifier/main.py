from utils.strategy import *
from utils.set_up import set_up

def run(ds:Dataset.DataSplits, methods_list, new_img_num_list, old_model_config:Config.OldModel, new_model_config:Config.NewModel, acquire_instruction:Config.Acquistion):
    for method in methods_list:
        strategy = StrategyFactory(method)
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items(method,new_img_num)
            strategy.operate(acquire_instruction, ds, old_model_config, new_model_config)

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', strategy='', device=0, augment=True):
    print('Use pure: ',pure)
    print('Use augment: ',augment)

    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, pure, device)

    method_list = ['dv','sm','conf','mix'] if strategy=='non_seq' else [strategy]
    # method_list = ['seq']
    # method_list, new_img_num_list = ['dv','sm'], [150,200]
    # new_img_num_list = [150,200,]
    for epo in range(epochs):
        print('In epoch {}'.format(epo))
        ds = ds_list[epo]
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter, augment)
        acquire_instruction = Config.AcquistionFactory(strategy,seq_rounds_config)
        run(ds,method_list, new_img_num_list, old_model_config,new_model_config, acquire_instruction)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-s','--strategy',type=str)
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-a','--augment',type=Config.str2bool, default=False)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,strategy=args.strategy,device=args.device, augment=args.augment)
    # print(args.method)