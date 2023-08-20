import tools
from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.checker as Checker
import utils.statistics.subset as Subset
import utils.statistics.stat_test as Plot_stat


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
    stream_instruction = Config.ProbabStream(bound=0.5, pdf='kde')

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = tools.get_probab_checker(epo, parse_para, ds_list[epo], clip_processor, stream_instruction, plot=False)
        data_splits = Dataset.DataSplits(ds_list[0], 10)
        market_dv, _ = checker.clf.predict(data_splits.loader['market'], checker.base_model)
        test_dv, _ = checker.clf.predict(data_splits.loader['test_shift'], checker.base_model)
        print(Plot_stat.kstest(market_dv, test_dv))
    

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device)