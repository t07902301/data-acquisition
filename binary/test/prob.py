from tools import epoch_run, get_probab_checker
from utils.strategy import *
from utils.set_up import set_up

def bound_run(epochs, parse_para, dataset_list, new_img_num_list, method_list, clip_processor, acquire_instruction:Config.Acquistion, stream_instruction:Config.Stream):
    results = []
    bound_stat_list = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = get_probab_checker(epo, parse_para, dataset_list[epo], clip_processor, stream_instruction)
        result_epoch, bound_stat = epoch_run(new_img_num_list, method_list, acquire_instruction, checker)
        results.append(result_epoch)
        bound_stat_list.append(bound_stat)
    return results, bound_stat_list


def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0):
    print('Use pure: ',pure)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    # method_list, method_labels = ['dv','sm','conf','seq_clf'], ['greedy decision value','random sampling','model confidence','sequential with only SVM updates']
    method_list, method_labels = ['dv'], ['dv']
    # method_list = ['dv','sm','conf','mix']
    # method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling']

    clip_processor = Detector.load_clip(device_config)
    parse_para = (batch_size, superclass_num, model_dir, device_config, pure, new_model_setter, seq_rounds_config)
    acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
    stream_instruction = Config.ProbabStream(bound=0.7, pdf='kde')
    # bounds = [-1.5, -1, -0.8, -0.5, 0, 0.5]
    bounds = [ None]
    # new_img_num_list = [100]
    bound_stat_list = []
    average_results = []
    for bound in bounds:
        print('In treshold: {}'.format(bound))
        acquire_instruction.bound = bound
        result, bound_stat = bound_run(epochs, parse_para, ds_list, new_img_num_list, method_list, clip_processor, acquire_instruction, stream_instruction)
        method_result = np.array(result)[:, 0, :]
        average_results.append(np.round(np.mean(method_result, axis=0), decimals=3).tolist())
        bound_stat_list.append(bound_stat)
   
    for i in average_results:
        print(*i, sep=',')

    # print(np.round(np.mean(bound_stat_list, axis=1), decimals=2))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device)