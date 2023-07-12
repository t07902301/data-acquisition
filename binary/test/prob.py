from tools import * 
from utils.set_up import set_up

def bound_run(epochs, parse_para, dataset_list, new_img_num_list, method_list, acquire_instruction:Config.Acquistion, stream_instruction:Config.Stream):
    results = []
    bound_stat_list = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = get_probab_checker(epo, parse_para, dataset_list[epo], stream_instruction, acquire_instruction.detector, plot=False)
        result_epoch, bound_stat = epoch_run(new_img_num_list, method_list, acquire_instruction, checker)
        results.append(result_epoch)
        bound_stat_list.append(bound_stat)
    return results, bound_stat_list


def main(epochs, new_model_setter='refine', pure=False, model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = ''):
    print('Detector:', detector_name)
    if pure == False:
        probab_bound = 0
    print('Probab bound:', probab_bound)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device)
    # method_list = ['dv','sm','conf'] if new_model_setter!='refine' else ['dv']
    method_list = ['dv','sm','conf']
    clip_processor = Detector.load_clip(device_config)
    parse_para = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
    acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde')
    detect_instruction = Config.Dectector(detector_name, clip_processor)
    acquire_instruction.add_detector(detect_instruction)
    # bounds = [-1.5, -1, -0.8, -0.5, 0, 0.5]
    bounds = [None]
    for bound in bounds:
        print('In treshold: {}'.format(bound))
        acquire_instruction.bound = bound
        result, bound_stat = bound_run(epochs, parse_para, ds_list, new_img_num_list, method_list, acquire_instruction, stream_instruction)
        result = np.array(result)
        print(result.shape)
        for idx, method in enumerate(method_list):
            method_result = result[:, idx, :]
            print(method)
            print(*np.round(np.mean(method_result, axis=0), decimals=3).tolist(), sep=',')

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name)