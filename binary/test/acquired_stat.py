import tools
from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.checker as Checker
import utils.statistics.subset as Subset
import utils.statistics.stat_test as Plot_stat

def n_data_run(acquisition_config, checker: Checker.subset, dataset, pdf_method):
    check_result = check_data(acquisition_config, dataset, checker, pdf_method)
    return check_result

def dv_dstr_plot(cor_dv, incor_dv, n_data, pdf_method=None, range=None):
    pdf_name = '' if pdf_method == None else '_{}'.format(pdf_method)
    Plot_stat.base_plot(cor_dv, 'correct', 'orange', pdf_method, range)
    Plot_stat.base_plot(incor_dv, 'incorrect', 'blue', pdf_method, range)
    Plot_stat.plt.savefig('figure/train/dv{}_{}.png'.format(pdf_name, n_data))
    Plot_stat.plt.close()
    print('Save fig to figure/train/dv{}_{}.png'.format(pdf_name, n_data))
    
def method_run(n_img_list, acquisition_config:Config.Acquistion, checker: Checker.subset, dataset, pdf_method):
    acc_change_list = []
    for n_img in n_img_list:
        acquisition_config.n_ndata = n_img
        acc_change = n_data_run(acquisition_config, checker, dataset, pdf_method)
        acc_change_list.append(acc_change)
    return acc_change_list

def run(acquisition_config:Config.Acquistion, methods, new_img_num_list, checker:Checker.subset, dataset, pdf_method):
    result_list = []
    for method in methods:
        print('In method', method)
        acquisition_config.method = method
        if 'seq' in method:
            checker.clf = Log.get_log_clf(acquisition_config, checker.model_config, checker.clip_set_up_loader, checker.clip_processor)
        result_method = method_run(new_img_num_list, acquisition_config, checker, dataset, pdf_method)
        result_list.append(result_method)
    return result_list

def epoch_run(new_img_num_list, method_list, acquire_instruction:Config.Acquistion, checker: Checker.subset, dataset, pdf_method):
    result_epoch = run(acquire_instruction, method_list, new_img_num_list, checker, dataset, pdf_method)
    return result_epoch

def check_data(acquisition_config:Config.Acquistion, dataset, checker:Checker.subset, pdf_method):
    model_config = checker.model_config
    model_config.set_path(acquisition_config)
    new_data = Log.get_log_data(acquisition_config, model_config, dataset)
    new_data_loader = torch.utils.data.DataLoader(new_data, batch_size= model_config.new_batch_size)
    cor_dv, incor_dv = Plot_stat.get_dv_dstr(checker.base_model, new_data_loader, checker.clf)
    plot_range = (-2.2312224442362742, -0.35740602488727813)
    dv_dstr_plot(cor_dv, incor_dv, acquisition_config.n_ndata, pdf_method, plot_range)
    total_dv = np.concatenate((cor_dv, incor_dv))
    dv_range = (min(total_dv), max(total_dv))
    print(dv_range)
    # ks_result = Plot_stat.kstest(cor_dv, Plot_stat.ecdf(incor_dv))
    # return ks_result.pvalue
    # incor_pred_stat = Subset.incorrect_pred_stat(new_data_loader, checker.base_model)
    # return incor_pred_stat


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

    results = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = tools.get_probab_checker(epo, parse_para, ds_list[epo], clip_processor, stream_instruction, plot=False)
        data_splits = Dataset.DataSplits(ds_list[0], 10)
        result_epoch = epoch_run(new_img_num_list, method_list, acquire_instruction, checker, data_splits.dataset['market'], stream_instruction.pdf)
        results.append(result_epoch)
    
    print(results)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device)