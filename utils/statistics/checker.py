import utils.objects.Config as Config
import utils.objects.Config as Config
import utils.objects.model as Model
import utils.objects.Detector as Detector
import utils.statistics.distribution as distribution_utils
from utils.objects.log import Log
import utils.objects.Config as Config
from abc import abstractmethod
import numpy as np
import torch
import utils.statistics.partitioner as Partitioner
import utils.statistics.data as DataStat
import utils.statistics.decision as Decision
import utils.objects.dataloader as dataloader_utils
from typing import Dict
from utils.logging import *
import utils.dataset.wrappers as dataset_utils
import utils.objects.utility_estimator as ue
from utils.parse_args import ParseArgs

class Prototype():
    '''
    Decide what to test the new and the old model
    '''
    def __init__(self, new_model_config:Config.NewModel, general_config) -> None:
        self.new_model_config = new_model_config
        self.general_config = general_config

    @abstractmethod
    def run(self, operation:Config.Operation):
        '''
        Load and test new model
        '''
        pass

    @abstractmethod
    def _target_test(self, loader, new_model: Model.Prototype):
        '''
        Test new model on given loaders
        '''
        pass

    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        '''
        Use the old model and data split to set up a Prototype (for each epoch) -> base_model, detector/detector, test data extracted info
        '''
        logger.info('Set up: ')
        self.vit = operation.detection.vit
        self.base_model = self.load_model(old_model_config)
        self.base_acc = self.base_model.acc(datasplits.loader['test_shift'])
        self.detector = Detector.factory(operation.detection.name, self.general_config, clip_processor = self.vit)
        self.detector.fit(self.base_model, datasplits.loader['val_shift'])

    def load_model(self, model_config: Config.Model, source=True):
        model = Model.factory(model_config.model_type, self.general_config, self.vit, source=source)
        model.load(model_config.path, model_config.device)
        return model
    
    def get_new_model(self, operation:Config.Operation):
        self.new_model_config.set_path(operation)
        new_model = self.load_model(self.new_model_config, source=False)
        return new_model

class Total(Prototype):
    '''
    Test New Model on the Entire Test Set
    '''
    def __init__(self, new_model_config: Config.NewModel, general_config) -> None:
        super().__init__(new_model_config, general_config)

    def set_up(self, old_model_config: Config.OldModel, datasplits: dataset_utils.DataSplits, operation: Config.Operation):
        logger.info('Set up: ')
        self.vit = operation.detection.vit
        self.base_model = self.load_model(old_model_config)
        self.base_acc = self.base_model.acc(datasplits.loader['test_shift'])
        self.test_loader = datasplits.loader['test_shift']

    def run(self, operation: Config.Operation):
        new_model = self.get_new_model(operation)
        return self._target_test(self.test_loader, new_model)
    
    def _target_test(self, loader, new_model:Model.Prototype):
        gt,pred,_  = new_model.eval(loader)
        total_correct = (gt==pred)
        # DataStat.evaluation_metric(loader['new_model'], self.base_model, new_model=new_model)
        return total_correct.mean()*100 - self.base_acc 

class Partition(Prototype):
    '''
    Split test set and then feed test splits into models;\n
    Only use model's hard label outputs
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)

    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.test_info = DataStat.Info(datasplits, 'test_shift', self.new_model_config.new_batch_size)

    def seq_set_up(self, operation:Config.Operation):
        '''
        Set up detector from sequential acquisition and anything that uses the detector
        '''
        logger.info('Seq Running:')
        detector_log = Log(self.new_model_config, 'detector')
        self.detector = detector_log.import_log(operation, self.general_config)

    @abstractmethod
    def run(self, operation:Config.Operation):
        pass

    @abstractmethod
    def stat_run(self, operation:Config.Operation):
        pass

    def error_stat(self, dataloader):
        '''
        How many errors of the old model in the new model test set?
        '''
        if len(dataloader['new_model']) == 0:
            new_error_stat = 0
            logger.info('No data to test the new model')
        else:
            new_error_stat = 100 - self.base_model.acc(dataloader['new_model'])
        return new_error_stat
    
    def _target_test(self, loader, new_model:Model.Prototype):
        '''
        loader: new_model + old_model
        '''

        if len(loader['old_model']) == 0:
            logger.info('Nothing for Old Model')
            gt,pred,_  = new_model.eval(loader['new_model'])
            new_correct = (gt==pred)
            total_correct = new_correct

        elif len(loader['new_model']) == 0:
            logger.info('Nothing for New Model')
            gt,pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (gt==pred)
            total_correct = old_correct

        else:
            new_gt, new_pred,_  = new_model.eval(loader['new_model'])
            new_correct = (new_gt==new_pred)
            old_gt, old_pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (old_gt==old_pred)
            total_correct = np.concatenate((old_correct,new_correct))
        
        # DataStat.evaluation_metric(loader['new_model'], self.base_model, new_model=new_model)

        return total_correct.mean()*100 - self.base_acc 
    
class DataValuation(Partition):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)

    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.set_utility_estimator(operation.acquisition.utility_estimation, operation.ensemble.pdf, datasplits)
        self.anchor_loader = datasplits.loader['val_shift'] # keep for Seq
        
    def set_utility_estimator(self, estimator_name, pdf, data_split:dataset_utils.DataSplits):
        self.utility_estimator = ue.factory(estimator_name, self.detector, data_split.loader['val_shift'], pdf, self.base_model)
        logger.info('Set up utility estimator.')

    def update_utility_estimator(self, operation:Config.Operation):
        self.utility_estimator = ue.factory(operation.acquisition.utility_estimation, self.detector, self.anchor_loader, operation.ensemble.name, self.base_model)
        logger.info('Update utility estimator.')

    def set_raw_data(self, dataset_name, data_config, normalize_stat):
        if dataset_name == 'cifar':
            self.data = dataset_utils.Cifar().get_raw_dataset(data_config['root'], normalize_stat, data_config['labels']['map'])['train_market']
        else:
            sampled_meta = dataset_utils.MetaData(data_config['root'])
            self.data = dataset_utils.Core().get_raw_dataset(sampled_meta, normalize_stat, data_config['labels']['map'])
    
    def get_subset_loader(self, criterion):
        loader = Partitioner.DataValuation().run(self.test_info, self.utility_estimator, criterion)
        return loader
    
    def import_new_data(self, operation:Config.Operation):
        log = Log(self.new_model_config, 'indices')
        new_data_indices = log.import_log(operation, self.general_config)
        new_data = torch.utils.data.Subset(self.data, new_data_indices)
        return new_data
    
    def stat_run(self, operation:Config.Operation):
        self.new_model_config.set_path(operation)
        new_data = self.import_new_data(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        criterion = self.get_criterion(new_data, self.new_model_config.new_batch_size)
        test_loader = self.get_subset_loader(criterion)
        return self.error_stat(test_loader)
    
    def seq_set_up(self, operation:Config.Operation):
        super().seq_set_up(operation)
        self.update_utility_estimator(operation)

    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        new_data = self.import_new_data(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        criterion = self.get_criterion(new_data, self.new_model_config.new_batch_size)
        test_loader = self.get_subset_loader(criterion)
        return self._target_test(test_loader, new_model)

    def get_criterion(self, dataset, batch_size):
        new_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return min(self.utility_estimator.run(new_data_loader))
    
class WeaknessScore(Partition):
    '''
    Feature Score as Distribution Probability
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.test_loader, _ = self.get_subset_loader(operation.ensemble)
     
    def get_subset_loader(self, ensemble_instruction:Config.Ensemble):
        loader, probabs = Partitioner.ProbabWeaknessScore().run(self.test_info, self.detector, ensemble_instruction)
        return loader, probabs

    def seq_set_up(self, operation:Config.Operation):
        super().seq_set_up(operation)
        self.test_loader, _ = self.get_subset_loader(operation.ensemble)

    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self._target_test(self.test_loader, new_model)
    
    def stat_run(self, operation:Config.Operation):
        self.new_model_config.set_path(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self.error_stat(self.test_loader)

class Posterior(Partition):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.anchor_loader = datasplits.loader['val_shift'] # keep for Seq
        weakness_score_dstr = self.set_up_dstr(self.anchor_loader, operation.ensemble.pdf)
        self.test_loader, posteriors = self.get_subset_loader(weakness_score_dstr, operation.ensemble)

        # fig_name = 'figure/test/probab_dstr.png' # w_p distribution between c_w and c_w'
        # self.probab_dstr_plot(posteriors, fig_name)
     
    def set_up_dstr(self, set_up_loader, pdf_type):
        correct_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, set_up_loader, pdf_type, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, set_up_loader, pdf_type, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}

    def get_subset_loader(self, weakness_score_dstr:Dict[str, distribution_utils.CorrectnessDisrtibution], ensemble_instruction:Config.Ensemble):
        loader, posteriors = Partitioner.Posterior().run(self.test_info, {'target': weakness_score_dstr['incorrect'], 'other': weakness_score_dstr['correct']}, ensemble_instruction, self.detector)
        return loader, posteriors
   
    def seq_set_up(self, operation:Config.Operation):
        super().seq_set_up(operation)
        weakness_score_dstr = self.set_up_dstr(self.anchor_loader, operation.ensemble.pdf)
        self.test_loader, posteriors = self.get_subset_loader(weakness_score_dstr, operation.ensemble)
       
        # fig_name = 'figure/test/probab_dstr_seq.png' # w_p distribution between c_w and c_w'
        # self.probab_dstr_plot(posteriors, fig_name)

    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self._target_test(self.test_loader, new_model)
    
    def stat_run(self, operation:Config.Operation):
        self.new_model_config.set_path(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self.error_stat(self.test_loader)
    
    def probab_dstr_plot(self, probab, fig_name, pdf_method=None):
        # distribution_utils.plt.hist(probab, bins=10)
        dataset_gts, dataset_preds, _ = self.base_model.eval(self.test_info.loader)
        correct_mask = (dataset_gts == dataset_preds)
        distribution_utils.base_plot(probab[correct_mask], 'C_w\'', 'white', pdf_method, hatch_style='/')        
        incorrect_mask = ~correct_mask
        distribution_utils.base_plot(probab[incorrect_mask], 'C_w', 'white', pdf_method, hatch_style='.', alpha=0.5)  
        distribution_utils.plt.xlabel('Weakness Probability f(x)', fontsize=12)
        distribution_utils.plt.ylabel('Probability Density', fontsize=12)
        distribution_utils.plt.xticks(fontsize=15)
        distribution_utils.plt.yticks(fontsize=15)
        distribution_utils.plt.savefig(fig_name)
        distribution_utils.plt.close()
        logger.info('Save fig to {}'.format(fig_name))  
        # logger.info('error with probab <= 0.5: {}'.format((probab[incorrect_mask] <= 0.5).sum())) 
        # logger.info('error with probab > 0.5: {}'.format((probab[incorrect_mask] > 0.5).sum())) 

class Ensemble(Prototype):
    '''
    Apply both the old and the new model to the entire test set;\n
    Use prediction probability from the model.
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.test_loader = datasplits.loader['test_shift']

    def ensemble_decision(self, new_probab, new_weights, old_probab, old_weights):
        '''
        Ensembled Decisions for Each Class
        '''
        return new_probab * new_weights + old_probab * old_weights
    
    @abstractmethod
    def get_weight(self, dataloader):
        pass
    
    def seq_set_up(self, operation:Config.Operation):
        '''
        Set up detector from sequential acquisition results and if anything that uses the detector
        '''
        logger.info('Seq Running:')
        detector_log = Log(self.new_model_config, 'detector')
        self.detector = detector_log.import_log(operation, self.general_config)

    @abstractmethod
    def run(self, operation:Config.Operation):
        pass
    
    def _target_test(self, dataloader, new_model: Model.Prototype):
     
        weights = self.get_weight(dataloader)

        decision_maker = Decision.factory(self.new_model_config.model_type, self.new_model_config.class_number)
        new_decision = decision_maker.get(new_model, dataloader)
        old_decision = decision_maker.get(self.base_model, dataloader)

        probab = self.ensemble_decision(new_decision, weights['new'], old_decision, weights['old'])
        decision = decision_maker.apply(probab)

        gts = dataloader_utils.get_labels(dataloader)
        final_acc = (gts==decision).mean() * 100 

        # DataStat.evaluation_metric(dataloader, self.base_model, ensemble_decision=decision)
        
        return final_acc - self.base_acc  

class WeaknessScoreEnsemble(Ensemble):
    '''
    Feature Score as Distribution Probability
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)

    def get_weight(self, dataloader):
        new_weight, _ = self.detector.predict(dataloader) 
        new_weight =  np.array(new_weight).reshape((len(new_weight),1))
        old_weight = 1 - new_weight
        return {
            'new': new_weight,
            'old': old_weight
        }
    
    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self._target_test(self.test_loader, new_model)
    
class PosteriorEnsemble(Ensemble):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def set_up(self, old_model_config:Config.OldModel, datasplits:dataset_utils.DataSplits, operation:Config.Operation):
        super().set_up(old_model_config, datasplits, operation)
        self.probab_partitioner = Partitioner.Posterior()
        self.anchor_loader = datasplits.loader['val_shift'] # keep for Seq
        self.weakness_score_dstr = self.set_up_dstr(self.anchor_loader, operation.ensemble.pdf)

    def set_up_dstr(self, set_up_loader, pdf_type):
        correct_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, set_up_loader, pdf_type, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, set_up_loader, pdf_type, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}
    
    def seq_set_up(self, operation: Config.Operation):
        super().seq_set_up(operation)
        self.weakness_score_dstr = self.set_up_dstr(self.anchor_loader, operation.ensemble.pdf)

    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        if 'seq' in operation.acquisition.method:
            self.seq_set_up(operation)
        return self._target_test(self.test_loader, new_model)
    
    def poterior2weight(self, dstr_dict: Dict[str, distribution_utils.CorrectnessDisrtibution], observations):
        return self.probab_partitioner.get_posterior(observations, dstr_dict)
    
    def get_weight(self, dataloader):
        weakness_score, _ = self.detector.predict(dataloader)  
        new_weight = self.poterior2weight({'target': self.weakness_score_dstr['incorrect'], 'other': self.weakness_score_dstr['correct']}, weakness_score)
        new_weight =  np.array(new_weight).reshape((len(new_weight),1))
        old_weight = 1 - new_weight
        # logger.info(new_weight.shape, old_weight.shape)
        return {
            'new': new_weight,
            'old': old_weight
        }
    
class AverageEnsemble(Ensemble):
    '''
    New and old model have the same credibility
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)

    def get_weight(self, dataloader):
        return {
            'new': 1,
            'old': 1
        }
    
    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        return self._target_test(self.test_loader, new_model)
    
# class AdaBoostEnsemble(Ensemble):
#     def __init__(self, model_config: Config.NewModel) -> None:
#         super().__init__(model_config)
#         self.old_alpha = self.get_boosting_alpha(self.base_model, self.anchor_loader)
    
#     def get_boosting_alpha(self, model:Model.Prototype, dataloader):
#         acc = model.acc(dataloader)
#         error_rate = 1 - acc
#         alpha = np.log(acc / error_rate) / 2
#         # gts, preds, _  = model.eval(dataloader)
#         # err_mask = (gts!=preds)
#         # total_err = err_mask.mean()
#         # alpha = np.log((1 - total_err) / total_err) / 2
#         # logger.info(total_err, alpha)
#         return alpha
    
#     def ensemble_probab(self, new_probab, old_probab, new_model, old_model, anchor_dataloader):

#         probab = new_probab * new_alpha + old_probab * old_alpha
#         return probab
    
def factory(name, new_model_config, general_config, use_posterior=True):
    if name == 'total':
        checker = Total(new_model_config, general_config)
    elif name == 'ae-c-wp':
        if use_posterior:
            checker = Posterior(new_model_config, general_config)
        else:
            checker = WeaknessScore(new_model_config, general_config)
    elif name == 'ae-w':
        if use_posterior:
            checker = PosteriorEnsemble(new_model_config, general_config)
        else:
            checker = WeaknessScoreEnsemble(new_model_config, general_config)
    elif name == 'avg-em':
        checker = AverageEnsemble(new_model_config, general_config)
    # elif name == 'ada':
    #     checker = AdaBoostEnsemble(new_model_config, general_config)
    elif name == 'ae-c-dv':
        checker = DataValuation(new_model_config, general_config)
    else:
        logger.info('Checker is not Implemented.')
        exit()
    return checker

def get_configs(epoch, parse_args:ParseArgs, dataset):
    old_model_config, new_model_config, general_config = Config.get_configs(epoch, parse_args)
    dataset_splits = dataset_utils.DataSplits(dataset, new_model_config.new_batch_size)
    return old_model_config, new_model_config, dataset_splits, general_config

def instantiate(epoch, parse_args, dataset, operation: Config.Operation, normalize_stat=None, dataset_name=None, use_posterior=True):
    old_model_config, new_model_config, dataset_splits, general_config = get_configs(epoch, parse_args, dataset)
    checker = factory(operation.ensemble.name, new_model_config, general_config, use_posterior)
    checker.set_up(old_model_config, dataset_splits, operation)
    if operation.ensemble.name == 'ae-c-dv':
        checker.set_raw_data(dataset_name, general_config['data'], normalize_stat)
    return checker
