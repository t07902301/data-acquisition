from utils.strategy import *
from utils.set_up import set_up

#TODO method -> strategy -> #new_data
# fix validation set to target test set before training new models
# copy dataset before manipulating

def run(operation: Config.Operation, old_model_config:Config.OldModel, new_model_config:Config.NewModel, dataset:dict):
    strategy = StrategyFactory(operation.acquisition.method, old_model_config)
    strategy.operate(operation, dataset, new_model_config)

def data_run(new_img_num_list, operation: Config.Operation, old_model_config:Config.OldModel, new_model_config:Config.NewModel, dataset:dict):
    for new_img_num in new_img_num_list:
        operation.acquisition.n_ndata = new_img_num
        run(operation, old_model_config, new_model_config, dataset)

def method_run(dataset:dict, methods_list, new_img_num_list, old_model_config:Config.OldModel, new_model_config:Config.NewModel, operation: Config.Operation):
    for method in methods_list:
        operation.acquisition.method = method
        operation.acquisition = Config.AcquisitionFactory(method, operation.acquisition)
        data_run(new_img_num_list, operation, old_model_config, new_model_config, dataset)

def epoch_run(parse_para, method_list, n_data_list, dataset, epo, operation: Config.Operation):
    batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config = parse_para
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
    method_run(dataset, method_list, n_data_list, old_model_config,new_model_config, operation)

def bound_run(parse_para, epochs, ds_list, method_list, new_img_num_list, bound, operation: Config.Operation):
    operation.acquisition.bound = bound
    for epo in range(epochs):
        print('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        epoch_run(parse_para, method_list, new_img_num_list, dataset, epo, operation)

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, base_type='', detector_name = ''):
    print('Detector:', detector_name)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device)

    probab_bound = 0.5 if pure else 0
    clip_processor = Detector.load_clip(device_config)
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)

    if new_model_setter!='refine':
        # method_list = ['dv','seq_clf']
        method_list = ['sm','conf']
    else:
        assert new_model_setter == 'refine' and pure == False
        method_list = ['dv']
    # method_list = ['dv','sm','conf', 'seq_clf'] 
    # method_list = ['dv']
    bounds = [None]
    for bound in bounds:
        parse_para = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
        bound_run(parse_para, epochs, ds_list, method_list, new_img_num_list, bound, operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-s','--strategy',type=str, default='non_seq')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-ns','--new_model_setter',type=str,default='retrain')

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, base_type=args.base_type, detector_name=args.detector_name, new_model_setter=args.new_model_setter)