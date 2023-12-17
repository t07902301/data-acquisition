from utils.strategy import *
from utils.set_up import *
import utils.statistics.checker as Checker
import utils.statistics.distribution as Distribution
from typing import List
from utils.logging import *
import utils.statistics.data as DataStat
import utils.statistics.distribution as distribution_utils

class ModelConf():
    def __init__(self) -> None:
        pass
    
    def budegt_run(self, budget_list, operation:Config.Operation,new_model_config:Config.NewModel, dataset_splits, config, weakness):
        results = []
        for budget in budget_list:
            operation.acquisition.set_budget(budget)
            new_model_config.set_path(operation)
            model = Model.factory(new_model_config.base_type, config, source=False)
            model.load(new_model_config.path, new_model_config.device)
            gts, preds, decision_scores = model.eval(dataset_splits.loader['val_shift'])
            targets = (gts!=preds) if weakness else (gts==preds)
            results.append(np.mean(acquistion.get_gt_probab(gts[targets], decision_scores[targets])))
        return results

    def run(self, epochs, parse_args, budget_list, operation:Config.Operation, dataset_list: List[dict], weakness):
        results = []
        model_dir, device_config, base_type, pure, new_model_setter, config = parse_args
        batch_size = config['hparams']['source']['batch_size']
        superclass_num = config['hparams']['source']['superclass']
        for epo in range(epochs):
            new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epo, pure, new_model_setter, batch_size['new'], base_type)
            logger.info('in epoch {}'.format(epo))
            dataset_splits = dataset_utils.DataSplits(dataset_list[epo], new_model_config.new_batch_size)
            budget_result = self.budegt_run(budget_list, operation, new_model_config, dataset_splits, config, weakness)
            results.append(budget_result)
        return results 

class BaseConf():
    def __init__(self) -> None:
        pass
    def run(self, epochs, parse_args, dataset_list, weakness): 
        results = []
        model_dir, device_config, base_type, pure, new_model_setter, config = parse_args
        batch_size = config['hparams']['source']['batch_size']
        superclass_num = config['hparams']['source']['superclass']
        for epo in range(epochs):
            old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epo, base_type)
            logger.info('in epoch {}'.format(epo))
            dataset_splits = dataset_utils.DataSplits(dataset_list[epo], old_model_config.batch_size)
            model = Model.factory(old_model_config.base_type, config)
            model.load(old_model_config.path, old_model_config.device)
            gts, preds, decision_scores = model.eval(dataset_splits.loader['val_shift'])
            targets = (gts!=preds) if weakness else (gts==preds)
            results.append(np.mean(acquistion.get_gt_probab(gts[targets], decision_scores[targets])))
        return results
    
class TestData():
    def __init__(self) -> None:
        pass
    
    def run(self, epochs, parse_args, budget_list, operation:Config.Operation, dataset_list: List[dict], normalize_stat, dataset_name, use_posterior, plot):
        '''
        plot: visualize weakness score distribution of validation set?\n
        '''
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation, normalize_stat, dataset_name, use_posterior) #probab / ensemble
            stat = self.method_run(budget_list, operation, checker)
            results.append(stat)

        if plot:
            self.plot(checker, dataset_list[epo], operation.ensemble.pdf)
        return results
    
    def method_run(self, budget_list, operation:Config.Operation, checker: Checker.Partition):
        stat_list = []
        if operation.acquisition.method != 'seq':
            budget_list = [600]
        for budget in budget_list:
            operation.acquisition.set_budget(budget)
            acc_change = self.budget_run(operation, checker)
            stat_list.append(acc_change)
        return stat_list

    def budget_run(self, operation:Config.Operation, checker: Checker.Partition):
        check_result = checker.stat_run(operation)
        return check_result
    
    def plot(self, checker: Checker.Partition, dataset, pdf):
        datasplits = dataset_utils.DataSplits(dataset, checker.new_model_config.new_batch_size)
        # fig_name = 'figure/test/val_dv.png'
        # self.naive_plot(datasplits.loader['val_shift'], fig_name, checker.detector)
        # fig_name = 'figure/test/market_dv.png'
        # self.naive_plot(datasplits.loader['market'], fig_name, checker.detector)
        fig_name = 'figure/test/anchor_dv.png'
        incor_dv = DataStat.get_correctness_weakness_score(checker.base_model, datasplits.loader['val_shift'], checker.detector, correctness=False)
        cor_dv = DataStat.get_correctness_weakness_score(checker.base_model, datasplits.loader['val_shift'], checker.detector, correctness=True)
        # logger.info('Incorrection DSTR - max: {}, min:{}'.format(max(incor_dv), min(incor_dv)))
        # logger.info('Correction DSTR - max: {}, min:{}'.format(max(cor_dv), min(cor_dv)))
        self.correctness_dstr_plot(cor_dv, incor_dv, fig_name, pdf)
        # fig_name = 'figure/test/test_dv.png'
        # incor_dv = DataStat.get_correctness_weakness_score(checker.base_model, datasplits.loader['test_shift'], checker.detector, correctness=False)
        # cor_dv = DataStat.get_correctness_weakness_score(checker.base_model, datasplits.loader['test_shift'], checker.detector, correctness=True)
        # self.correctness_dstr_plot(cor_dv, incor_dv, fig_name, pdf)

    def correctness_dstr_plot(self, cor_dv, incor_dv, fig_name, pdf_method=None):
        distribution_utils.base_plot(cor_dv, 'C_w\'', 'white', pdf_method, hatch_style='/')
        distribution_utils.base_plot(incor_dv, 'C_w', 'white', pdf_method, hatch_style='.', line_style=':', alpha=0.5)
        distribution_utils.plt.xlabel('Weakness Score', fontsize=15)
        distribution_utils.plt.ylabel('Probability Density', fontsize=15)
        distribution_utils.plt.xticks(fontsize=15)
        distribution_utils.plt.yticks(fontsize=15)

        # distribution_utils.plt.title('Model Performance Feature Score Distribution')
        distribution_utils.plt.savefig(fig_name)
        distribution_utils.plt.close()
        logger.info('Save fig to {}'.format(fig_name))

    def naive_plot(self, dataloader, fig_name, detector: Detector.Prototype):
        weakness_score, _ = detector.predict(dataloader)
        logger.info('max: {}, min:{}'.format(max(weakness_score), min(weakness_score)))
        distribution_utils.base_plot(weakness_score, 'all data', 'orange', pdf_method='kde')
        distribution_utils.plt.xlabel('Feature Score')
        distribution_utils.plt.ylabel('Density')
        distribution_utils.plt.title('Old Model Performance Feature Score Distribution')
        distribution_utils.plt.savefig(fig_name)
        distribution_utils.plt.close()
        logger.info('Save fig to {}'.format(fig_name))

class TrainData():
    def __init__(self, dataset_name, data_config, normalize_stat) -> None:
        if dataset_name == 'cifar':
            self.data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            self.data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])
   
    def set_utility_estimator(self, detector_instruction: Config.Detection, estimator_name, pdf, general_config, data_split, base_model):
        detector = Detector.factory(detector_instruction.name, general_config, detector_instruction.vit)
        detector.fit(base_model,data_split.loader['val_shift'])
        self.utility_estimator = ue.factory(estimator_name, detector,data_split.loader['val_shift'], pdf, base_model)
        logger.info('Set up utility estimator.')

    def dv_dstr_plot(self, cor_dv, incor_dv, budget, pdf_method=None, range=None):
        pdf_name = '' if pdf_method == None else '_{}'.format(pdf_method)
        Distribution.base_plot(cor_dv, 'correct', 'orange', pdf_method, range)
        Distribution.base_plot(incor_dv, 'incorrect', 'blue', pdf_method, range)
        Distribution.plt.savefig('figure/train/weakness_score{}_{}.png'.format(pdf_name, budget))
        Distribution.plt.close()
        logger.info('Save fig to figure/train/weakness_score{}_{}.png'.format(pdf_name, budget))

    def run(self, epochs, parse_args, budget_list, operation:Config.Operation, dataset_list: List[dict]):
        results = []
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation)
            data_split = dataset_utils.DataSplits(dataset_list[epo], checker.general_config['hparams']['source']['batch_size']['new'])
            self.set_utility_estimator(operation.detection, operation.acquisition.utility_estimation, operation.ensemble.pdf, checker.general_config, data_split, checker.base_model)
            result_epoch = self.epoch_run(operation, budget_list, checker, data_split)
            results.append(result_epoch)
        return results

    def epoch_run(self, operation:Config.Operation, budget_list, checker:Checker.Prototype, data_split:dataset_utils.DataSplits):
        return self.method_run(budget_list, operation, checker, data_split)

    def method_run(self, budget_list, operation:Config.Operation, checker: Checker.Prototype, data_split:dataset_utils.DataSplits):
        acc_change_list = []
        for budget in budget_list:
            operation.acquisition.set_budget(budget)
            acc_change = self.budget_run(operation, checker, data_split)
            acc_change_list.append(acc_change)
        return acc_change_list
    
    def get_dataset_size(self, dataset):
        return len(dataset)
    
    def utility_range(self, dataset, batch_size):
        new_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # probab = distribution.pdf.accumulate(max(utility), min(utility))

        return min(self.utility_estimator.run(new_data_loader))
    
    def misclassifications(self, dataset, batch_size, base_model: Model.Prototype):
        new_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return 100 - base_model.acc(new_data_loader)
    
    def check_overfit(self, operation:Config.Operation, checker:Checker.Prototype):
        model_config = checker.new_model_config
        model_config.set_path(operation)
        log = Log(model_config, 'detector')
        detector = log.import_log(operation, checker.general_config)

        log = Log(model_config, 'indices')
        new_data_indices = log.import_log(operation, checker.general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices)

        new_loader = torch.utils.data.DataLoader(new_data, batch_size=checker.new_model_config.new_batch_size)
        _, metrics = detector.predict(new_loader, checker.base_model, 'prec')
        return metrics

    def check_indices(self, operation:Config.Operation, data_split:dataset_utils.DataSplits, checker:Checker.Prototype, distribution: distribution_utils.Disrtibution):
        model_config = checker.new_model_config
        model_config.set_path(operation)

        log = Log(model_config, 'indices')
        new_data_indices = log.import_log(operation, checker.general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices)

        # return self.get_dataset_size(new_data)
        # return self.utility_range(new_data, model_config.new_batch_size)
    
        # return self.misclassifications(new_data, model_config.new_batch_size, checker.base_model) * len(new_data) / 100
        return self.misclassifications(new_data, model_config.new_batch_size, checker.base_model)
        
    def check_clf(self, operation:Config.Operation, data_split:dataset_utils.DataSplits, checker:Checker.Prototype):
        model_config = checker.new_model_config
        model_config.set_path(operation)
        log = Log(model_config, 'detector')
        detector = log.import_log(operation, checker.general_config)
        _, metrics = detector.predict(data_split.loader['test_shift'], checker.base_model, metrics='prec')
        return metrics

    def budget_run(self, operation:Config.Operation, checker: Checker.Prototype, data_split:dataset_utils.DataSplits):
        # check_result = self.check_indices(operation, data_split, checker, None)
        # check_result = self.check_clf(operation, data_split, checker)
        check_result = self.check_overfit(operation, checker)
        return check_result
   
def main(epochs, new_model_setter='retrain', model_dir ='', device=0, base_type='', detector_name = '', mode='train', acquisition_method= 'weakness_score', ensemble_name=None, ensemble_criterion=None, pdf='kde', weakness=0, use_posterior=1, utility_estimator='u-ws'):
    pure = True
    weakness = True if weakness == 1 else False
    fh = logging.FileHandler('log/{}/stat_{}_{}.log'.format(model_dir, acquisition_method, mode), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Ensemble Name: {}, criterion:{}, use_posterior: {}, utility_estimator: {}'.format(ensemble_name, ensemble_criterion, use_posterior, utility_estimator))

    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(name=ensemble_name, criterion=ensemble_criterion)
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method=acquisition_method, data_config=config['data'], utility_estimator=utility_estimator)
    
    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
    
    if mode == 'train':
        stat_checker = TrainData(dataset_name, config['data'], normalize_stat)
        results = stat_checker.run(epochs, parse_args, config['data']['budget'], operation, ds_list)
        results = np.array(results)
        logger.info('Train Data stat: {}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))
        logger.info('all: {}'.format(results.tolist()))
    elif mode == 'conf':
        stat_checker = ModelConf()
        results = stat_checker.run(epochs, parse_args, config['data']['budget'], operation, ds_list, weakness=weakness)
        logger.info('Model Confidence stat:{}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))
    elif mode == 'bc':
        stat_checker = BaseConf()
        results = stat_checker.run(epochs, parse_args, ds_list, weakness=weakness)
        logger.info('Base Model Confidence stat:{}'.format(np.round(np.mean(results), decimals=3)))
        # logger.info('all: {}'.format(np.round(results, decimals=3).tolist()))
    else:
        stat_checker = TestData()
        results = stat_checker.run(epochs, parse_args, config['data']['budget'], operation, ds_list, normalize_stat, dataset_name, use_posterior, plot=False)
        logger.info('Test Data error stat:{}'.format(np.round(np.mean(results, axis=0), decimals=3).tolist()))
        # logger.info('all: {}'.format(results))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name)_task_(other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, regression; (regression: logistic regression)")
    parser.add_argument('-am','--acquisition_method',type=str, default='dv', help="Acquisition Strategy; dv:one-shot, rs: random, conf: Probability-at-Ground-Truth, mix: Random Weakness, seq: sequential, pd: one-shot + u-wsd, seq_pd: seq + u-wsd")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-mode','--mode',type=str, default='train', help="test, train: check statistics (e.g. misclassification percentage, final detector recall) of train or test data")
    parser.add_argument('-ec','--ensemble_criterion',type=float,default=0.5, help='A threshold of the probability from Cw to assign test set and create corresponding val set for model training.')
    parser.add_argument('-em','--ensemble',type=str, default='total', help="Ensemble Method")
    parser.add_argument('-w','--weakness',type=int, default=0, help="check weakness or not")
    parser.add_argument('-up','--use_posterior',type=str2bool, default=1, help="use posterior or not")
    parser.add_argument('-ue','--utility_estimator',type=str, default='u-ws', help="u-ws, u-wsd")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, base_type=args.base_type, detector_name=args.detector_name, acquisition_method=args.acquisition_method, mode=args.mode, ensemble_criterion=args.ensemble_criterion, ensemble_name=args.ensemble, weakness = args.weakness, use_posterior=args.use_posterior, utility_estimator=args.utility_estimator)
