from utils.strategy import *
from utils.set_up import set_up

def n_data_run(acquire_instruction: Config.Acquistion, old_model_config:Config.OldModel, new_model_config:Config.NewModel, dataset):
    strategy = StrategyFactory(acquire_instruction.method, old_model_config)
    strategy.operate(acquire_instruction, dataset, new_model_config)

def method_run(new_img_num_list, acquire_instruction: Config.Acquistion, old_model_config:Config.OldModel, new_model_config:Config.NewModel, dataset):
    for new_img_num in new_img_num_list:
        acquire_instruction.n_ndata = new_img_num
        n_data_run(acquire_instruction, old_model_config, new_model_config, dataset)

def run(dataset:dict, methods_list, new_img_num_list, old_model_config:Config.OldModel, new_model_config:Config.NewModel, acquire_instruction:Config.Acquistion):
    for method in methods_list:
        acquire_instruction.method = method
        method_run(new_img_num_list, acquire_instruction, old_model_config, new_model_config, dataset)

def epoch_run(parse_para, method_list, n_data_list, dataset, acquire_instruction, epo):
    batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config = parse_para
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
    run(dataset, method_list, n_data_list, old_model_config,new_model_config, acquire_instruction)

def bound_run(parse_para, epochs, ds_list, strategy:str, method_list, new_img_num_list, bound):
    acquire_instruction = Config.AcquistionFactory(strategy, sequential_rounds_config=0)
    acquire_instruction.bound = bound
    for epo in range(epochs):
        print('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        epoch_run(parse_para, method_list, new_img_num_list, dataset, acquire_instruction, epo)

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', strategy='', device=0, base_type=''):
    print('Use pure: ',pure)

    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)

    # method_list = ['dv','sm','conf','mix'] if strategy=='non_seq' else [strategy]
    method_list = ['dv']
    bounds = [-1.5, -1, -0.8, -0.5, 0, 0.5]
    # bounds = [ -0.5, 0, 0.5]


    # method_list, new_img_num_list = ['dv','sm'], [150,200]
    # new_img_num_list = [150,200,]
    bounds = [None]
    for bound in bounds:
        parse_para = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
        bound_run(parse_para, epochs, ds_list, strategy, method_list, new_img_num_list, bound)



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-s','--strategy',type=str)
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='resnet')

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,strategy=args.strategy,device=args.device, base_type=args.base_type)
    # print(args.method)