from utils.strategy import *
from utils.set_up import *

def run(operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    strategy = StrategyFactory(operation.acquisition.method)
    strategy.operate(operation, new_model_config, workspace)

def data_run(new_img_num_list, operation: Config.Operation, new_model_config:Config.NewModel, workspace: WorkSpace):
    for new_img_num in new_img_num_list:
        operation.acquisition.n_ndata = new_img_num
        run(operation, new_model_config, workspace)

def method_run(methods_list, new_img_num_list, new_model_config:Config.NewModel, operation: Config.Operation, workspace: WorkSpace):
    for method in methods_list:
        operation.acquisition.method = method
        operation.acquisition = Config.AcquisitionFactory(method, operation.acquisition)
        data_run(new_img_num_list, operation, new_model_config, workspace)

def epoch_run(parse_para, method_list, n_data_list, dataset:dict, epo, operation: Config.Operation):
    batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config = parse_para
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
    workspace = WorkSpace(old_model_config, dataset)

    print('Set up WorkSpace')
    
    workspace.set_up(new_model_config.new_batch_size, operation.detection.vit)
    workspace.set_validation(operation.stream, old_model_config.batch_size, new_model_config.new_batch_size, detect_instruction=operation.detection)
    method_run(method_list, n_data_list, new_model_config, operation, workspace)

def bound_run(parse_para, epochs, ds_list, method_list, new_img_num_list, bound, operation: Config.Operation):
    operation.acquisition.bound = bound
    for epo in range(epochs):
        print('In epoch {}'.format(epo))
        dataset = ds_list[epo]
        epoch_run(parse_para, method_list, new_img_num_list, dataset, epo, operation)

def dev(epochs, dev_name, device, detector_name, model_dir, base_type):
    pure, new_model_setter = True, 'retrain'
    if dev_name == 'sm':
        method_list, probab_bound = [dev_name], 0
    elif dev_name == 'refine':
        method_list, new_model_setter, pure, probab_bound = ['dv'], 'refine', False, 0
    else:
        # method_list, probab_bound = ['conf'], 0.5 
        method_list, probab_bound = [dev_name], 0.5 
        # method_list, probab_bound = ['seq_clf'], 0.5 

    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, new_img_num_list, superclass_num, seq_rounds_config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device)
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)

    parse_para = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
    bound_run(parse_para, epochs, ds_list, method_list, new_img_num_list, None, operation)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-bt','--base_type',type=str,default='cnn')

    args = parser.parse_args()
    dev(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, dev_name=args.dev, base_type=args.base_type)

# def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, base_type='', detector_name = ''):
#     print('Detector:', detector_name)
#     device_config = 'cuda:{}'.format(device)
#     torch.cuda.set_device(device_config)
#     batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device)

#     probab_bound = 0.5 if pure else 0
#     if new_model_setter!='refine':
#         # method_list = ['dv','seq_clf']
#         method_list = ['sm','conf']
#     else:
#         assert new_model_setter == 'refine' and pure == False
#         method_list = ['dv']

#     clip_processor = Detector.load_clip(device_config)
#     stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
#     detect_instruction = Config.Detection(detector_name, clip_processor)
#     acquire_instruction = Config.Acquisition()
#     operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)

#     bounds = [None]
#     for bound in bounds:
#         parse_para = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
#         bound_run(parse_para, epochs, ds_list, method_list, new_img_num_list, bound, operation)

# import argparse
# if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-e','--epochs',type=int,default=1)
    # parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    # parser.add_argument('-md','--model_dir',type=str,default='')
    # parser.add_argument('-s','--strategy',type=str, default='non_seq')
    # parser.add_argument('-d','--device',type=int,default=0)
    # parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    # parser.add_argument('-dn','--detector_name',type=str,default='svm')
    # parser.add_argument('-ns','--new_model_setter',type=str,default='retrain')

    # args = parser.parse_args()
    # main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, base_type=args.base_type, detector_name=args.detector_name, new_model_setter=args.new_model_setter)