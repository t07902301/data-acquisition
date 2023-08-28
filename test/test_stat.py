import sys
sys.path.append('/home/yiwei/data-acquisition/core/')
from utils.strategy import *
from utils.set_up import *
import utils.statistics.checker as Checker
import utils.statistics.distribution as Distribution

def get_errors(model: Model.Prototype, dataloader):
    '''
    Model Errors on a dataloader
    '''
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    error_mask = (dataset_gts != dataset_preds)
    return dataset_preds[error_mask]

def dv_dstr_plot(cor_dv, incor_dv, n_data, pdf_method=None, range=None):
    pdf_name = '' if pdf_method == None else '_{}'.format(pdf_method)
    Distribution.base_plot(cor_dv, 'correct', 'orange', pdf_method, range)
    Distribution.base_plot(incor_dv, 'incorrect', 'blue', pdf_method, range)
    Distribution.plt.savefig('figure/train/dv{}_{}.png'.format(pdf_name, n_data))
    Distribution.plt.close()
    print('Save fig to figure/train/dv{}_{}.png'.format(pdf_name, n_data))

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = '', opion = '', dataset_name = ''):
    print('Detector:', detector_name)
    # if pure == False:
    #     probab_bound = 0
    print('Probab bound:', probab_bound)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device, opion, dataset_name)
    # method_list = ['dv','sm','conf'] if new_model_setter!='refine' else ['dv']
    # method_list = [ 'dv', 'seq_clf']
    method_list = ['dv']

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)

    results = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        checker = Checker.instantiate(epo, parse_args, ds_list[epo], operation, plot=False)
        if len(checker.test_loader['new_model']) == 0:
            results.append(0)
        else:
            test_data_size = dataloader_utils.get_size(checker.test_loader['new_model'])
            error = get_errors(checker.base_model, checker.test_loader['new_model'])
            results.append(len(error) / test_data_size  * 100)
    # print(results)
    print(np.round(np.mean(results), decimals=3))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-ds','--dataset',type=str, default='core')
    parser.add_argument('-op','--option',type=str, default='object')

    args = parser.parse_args()
    main(args.epochs,pure=args.pure,model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name, opion=args.option, dataset_name=args.dataset)
