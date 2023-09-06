from utils.strategy import *
from utils.set_up import *
import utils.statistics.checker as Checker
import utils.statistics.distribution as Distribution
from typing import List
from utils.logging import *

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
    logger.info('Save fig to figure/train/dv{}_{}.png'.format(pdf_name, n_data))

class TestData():
    def __init__(self) -> None:
        pass

    def run(self, epochs, parse_args, ds_list, operation):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, ds_list[epo], operation, plot=False, error_stat=True)
            if len(checker.test_loader['new_model']) == 0:
                results.append(0)
            else:
                test_data_size = dataloader_utils.get_size(checker.test_loader['new_model'])
                error = get_errors(checker.base_model, checker.test_loader['new_model'])
                results.append(len(error) / test_data_size  * 100)
        return results

class TrainData():
    def __init__(self) -> None:
        pass

    def run(self, epochs, parse_args, new_img_num_list, method_list, operation:Config.Operation, dataset_list: List[dict], pdf_method):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, plot=False, error_stat=True)
            data_split = dataset_utils.DataSplits(dataset_list[epo], checker.general_config['hparams']['batch_size']['new'])
            result_epoch = self.epoch_run(operation, method_list, new_img_num_list, checker, data_split, pdf_method)
            results.append(result_epoch)
        return results

    def epoch_run(self, operation:Config.Operation, method, new_img_num_list, checker:Checker.Prototype, data_split:dataset_utils.DataSplits, pdf_method):
        logger.info('In method: {}'.format(method))
        operation.acquisition.method = method
        return self.method_run(new_img_num_list, operation, checker, data_split, pdf_method)

    def method_run(self, n_img_list, operation:Config.Operation, checker: Checker.Prototype, data_split:dataset_utils.DataSplits, pdf_method):
        acc_change_list = []
        for n_img in n_img_list:
            operation.acquisition.n_ndata = n_img
            acc_change = self.n_data_run(operation, checker, data_split, pdf_method)
            acc_change_list.append(acc_change)
        return acc_change_list

    def check_data(self, operation:Config.Operation, data_split:dataset_utils.DataSplits, checker:Checker.Prototype):
        model_config = checker.new_model_config
        model_config.set_path(operation)
        log = Log(model_config, 'data')
        new_data = log.import_log(operation, checker.general_config)
        new_data_loader = torch.utils.data.DataLoader(new_data, batch_size= model_config.new_batch_size)
        new_data_size = dataloader_utils.get_size(new_data_loader)
        errors = get_errors(checker.base_model, new_data_loader)
        return len(errors) / new_data_size * 100

    def check_indices(self, operation:Config.Operation, data_split:dataset_utils.DataSplits, checker:Checker.Prototype, pdf_method, distribution: distribution_utils.Disrtibution):
        model_config = checker.new_model_config
        model_config.set_path(operation)

        log = Log(model_config, 'indices')
        new_data_indices = log.import_log(operation, checker.general_config)
        new_data = torch.utils.data.Subset(data_split.dataset['market'], new_data_indices)
        new_data_loader = torch.utils.data.DataLoader(new_data, batch_size= model_config.new_batch_size)

        # new_data_dv, _ = checker.detector.predict(new_data_loader, checker.base_model)

        # probab = distribution.pdf.accumulate(max(new_data_dv), min(new_data_dv))

        # return probab

        new_data_size = dataloader_utils.get_size(new_data_loader)
        
        # old_labels = set(Subset.config['data']['train_label']) - set(Subset.config['data']['remove_fine_labels'])
        # logger.info(Subset.label_stat(new_data, Subset.config['data']['remove_fine_labels']), Subset.label_stat(new_data, old_labels))
        errors = get_errors(checker.base_model, new_data_loader)

        return len(errors) / new_data_size * 100
    
        # train_loader = torch.utils.data.DataLoader(data_split.dataset['train'], batch_size= model_config.new_batch_size)
        # base_gt, base_pred, _ = checker.base_model.eval(train_loader)
        # base_incor_mask = (base_gt != base_pred)
        # base_incor = base_gt[base_incor_mask]
        # return (len(incor) + len(base_incor)) / (acquisition_config.n_ndata + len(data_split.dataset['train'])) * 100

        # cor_dv, incor_dv = Distribution.get_dv_dstr(checker.base_model, new_data_loader, checker.detector)
        # logger.info('Old model mistakes in acquired data: {}%'.format())
        # plot_range = (-2.5, 0) # test_dv
        # dv_dstr_plot(cor_dv, incor_dv, acquisition_config.n_ndata, pdf_method, plot_range)

        # market_dv, _ = checker.detector.predict(data_split.loader['market'], checker.base_model)
        # test_dv, _ = checker.detector.predict(data_split.loader['test_shift'], checker.base_model)
        # new_data_dv = market_dv[indices]
        # new_data_dv, _ = checker.detector.predict(new_data_loader, checker.base_model)
        # ks_result = Distribution.kstest(new_data_dv, test_dv)
        # return ks_result.pvalue
        
    def check_clf(self, operation:Config.Operation, data_split:dataset_utils.DataSplits, checker:Checker.Prototype):
        model_config = checker.new_model_config
        model_config.set_path(operation)
        log = Log(model_config, 'detector')
        detector = log.import_log(operation, checker.general_config)
        _, prec = detector.predict(data_split.loader['test_shift'], checker.base_model, compute_metrics=True)
        return prec

    def n_data_run(self, operation:Config.Operation, checker: Checker.Prototype, data_split:dataset_utils.DataSplits, pdf_method):
        if 'seq' in operation.acquisition.method:
            # check_result = self.check_data(operation, data_split, checker)
            check_result = self.check_clf(operation, data_split, checker)
        else:
            distribution = distribution_utils.Disrtibution(checker.detector, data_split.loader['val_shift'], operation.stream.pdf)
            check_result = self.check_indices(operation, data_split, checker, pdf_method, distribution)
        return check_result

def main(epochs, new_model_setter='retrain', model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = '', opion = '', dataset_name = '', stat_data='train', dev_name= 'dv'):
    pure = True
    fh = logging.FileHandler('log/{}/stat_{}_{}.log'.format(model_dir, dev_name, stat_data),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Detector: {}; Probab bound: {}'.format(detector_name, probab_bound))
    
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device, opion, dataset_name)
    method = dev_name

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
    
    if stat_data == 'train':
        stat_checker = TrainData()
        results = stat_checker.run(epochs, parse_args, config['data']['n_new_data'], method, operation, ds_list, stream_instruction.pdf)
        results = np.array(results)
        logger.info(method)
        logger.info('Train Data stat: {}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))
        logger.info('all: {}'.format(results.tolist()))
    else:
        stat_checker = TestData()
        results = stat_checker.run(epochs, parse_args, ds_list, operation)
        logger.info('Test Data error stat:{}'.format(np.round(np.mean(results), decimals=3)))
        logger.info('all: {}'.format(results))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-ds','--dataset',type=str, default='core')
    parser.add_argument('-op','--option',type=str, default='object')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-stat','--stat',type=str, default='train')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name, opion=args.option, dataset_name=args.dataset, dev_name=args.dev, stat_data=args.stat)
