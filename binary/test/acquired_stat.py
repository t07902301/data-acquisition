import sys
sys.path.append('/home/yiwei/data-acquisition/binary/')
from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.checker as Checker
import utils.statistics.distribution as Distribution

def get_cor_incor_dstr(model: Model.prototype, dataloader):
    '''
    Correctly Classified Points, Misclassified Points
    '''
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    return dataset_gts[cor_mask], dataset_preds[incor_mask]

def n_data_run(operation:Config.Operation, checker: Checker.prototype, data_split:Dataset.DataSplits, pdf_method):
    if 'seq' in operation.acquisition.method:
        check_result = check_clf(operation, data_split, checker)
    else:
        check_result = check_data(operation, data_split, checker, pdf_method)
    return check_result

def dv_dstr_plot(cor_dv, incor_dv, n_data, pdf_method=None, range=None):
    pdf_name = '' if pdf_method == None else '_{}'.format(pdf_method)
    Distribution.base_plot(cor_dv, 'correct', 'orange', pdf_method, range)
    Distribution.base_plot(incor_dv, 'incorrect', 'blue', pdf_method, range)
    Distribution.plt.savefig('figure/train/dv{}_{}.png'.format(pdf_name, n_data))
    Distribution.plt.close()
    print('Save fig to figure/train/dv{}_{}.png'.format(pdf_name, n_data))
    
def method_run(n_img_list, operation:Config.Operation, checker: Checker.prototype, data_split:Dataset.DataSplits, pdf_method):
    acc_change_list = []
    for n_img in n_img_list:
        operation.acquisition.n_ndata = n_img
        acc_change = n_data_run(operation, checker, data_split, pdf_method)
        acc_change_list.append(acc_change)
    return acc_change_list

def run(operation:Config.Operation, methods, new_img_num_list, checker:Checker.prototype, data_split:Dataset.DataSplits, pdf_method):
    result_list = []
    for method in methods:
        print('In method', method)
        operation.acquisition.method = method
        result_method = method_run(new_img_num_list, operation, checker, data_split, pdf_method)
        result_list.append(result_method)
    return result_list

def epoch_run(new_img_num_list, method_list, operation:Config.Operation, checker: Checker.prototype, data_split:Dataset.DataSplits, pdf_method):
    result_epoch = run(operation, method_list, new_img_num_list, checker, data_split, pdf_method)
    return result_epoch

def check_data(operation:Config.Operation, data_split:Dataset.DataSplits, checker:Checker.prototype, pdf_method):
    model_config = checker.model_config
    model_config.set_path(operation)

    log = Log(model_config, 'indices')
    new_data_indices = log.import_log(operation)
    new_data = torch.utils.data.Subset(data_split.dataset['market'], new_data_indices)
    new_data_loader = torch.utils.data.DataLoader(new_data, batch_size= model_config.new_batch_size)
    
    # old_labels = set(Subset.config['data']['train_label']) - set(Subset.config['data']['remove_fine_labels'])
    # print(Subset.label_stat(new_data, Subset.config['data']['remove_fine_labels']), Subset.label_stat(new_data, old_labels))
    cor, incor = get_cor_incor_dstr(checker.base_model, new_data_loader)
    return len(incor) / (len(cor) + len(incor)) * 100
   
    # train_loader = torch.utils.data.DataLoader(data_split.dataset['train'], batch_size= model_config.new_batch_size)
    # base_gt, base_pred, _ = checker.base_model.eval(train_loader)
    # base_incor_mask = (base_gt != base_pred)
    # base_incor = base_gt[base_incor_mask]
    # return (len(incor) + len(base_incor)) / (acquisition_config.n_ndata + len(data_split.dataset['train'])) * 100

    # cor_dv, incor_dv = Distribution.get_dv_dstr(checker.base_model, new_data_loader, checker.clf)
    # print('Old model mistakes in acquired data: {}%'.format())
    # plot_range = (-2.5, 0) # test_dv
    # dv_dstr_plot(cor_dv, incor_dv, acquisition_config.n_ndata, pdf_method, plot_range)

    # market_dv, _ = checker.clf.predict(data_split.loader['market'], checker.base_model)
    # test_dv, _ = checker.clf.predict(data_split.loader['test_shift'], checker.base_model)
    # new_data_dv = market_dv[indices]
    # new_data_dv, _ = checker.clf.predict(new_data_loader, checker.base_model)
    # ks_result = Distribution.kstest(new_data_dv, test_dv)
    # return ks_result.pvalue
    
def check_clf(operation:Config.Operation, data_split:Dataset.DataSplits, checker:Checker.prototype):
    model_config = checker.model_config
    model_config.set_path(operation)
    log = Log(model_config, 'clf')
    detector = log.import_log(operation)
    _, prec = detector.predict(data_split.loader['test_shift'], checker.base_model, compute_metrics=True)
    # _, prec = detector.predict(data_split.loader['val_shift'], checker.base_model, compute_metrics=True)
    return prec

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = ''):
    print('Detector:', detector_name)
    # if pure == False:
    #     probab_bound = 0
    print('Probab bound:', probab_bound)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device)
    # method_list = ['dv','sm','conf'] if new_model_setter!='refine' else ['dv']
    method_list = ['dv','sm','conf', 'seq_clf'] 
    # method_list = ['dv']

    clip_processor = Detector.load_clip(device_config)
    parse_args = (batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config)
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    results = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, ds_list[epo], operation, plot=False)
        data_splits = Dataset.DataSplits(ds_list[0], 10)
        result_epoch = epoch_run(new_img_num_list, method_list, operation, checker, data_splits, stream_instruction.pdf)
        results.append(result_epoch)
    
    result = np.array(results)
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
    parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name)
