import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import utils.objects.Config as Config
import utils.objects.model as Model
import utils.objects.Detector as Detector
import utils.objects.dataset as Dataset
import utils.statistics.distribution as Distribution
from utils.objects.log import Log
import utils.objects.Config as Config
from abc import abstractmethod
import numpy as np
import torch
import utils.statistics.partitioner as Partitioner
import utils.statistics.data as DataStat
import utils.statistics.decision as Decision
import utils.objects.dataloader as dataloader_utils

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
    def _target_test(self, loader, new_model):
        '''
        Test new model on given loaders
        '''
        pass

    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        '''
        Use the old model and data split to set up a Prototype (for each epoch) -> base_model, clf/detector, test data extracted info
        '''
        self.vit = operation.detection.vit
        self.base_model = self.load_model(old_model_config)
        self.base_acc = self.base_model.acc(datasplits.loader['test_shift'])
        self.clf = Detector.factory(operation.detection.name, self.general_config, clip_processor = self.vit)
        self.clf.fit(self.base_model, datasplits.loader['val_shift'])

    def load_model(self, model_config: Config.Model):
        model = Model.factory(model_config.base_type, self.general_config, self.vit)
        model.load(model_config.path, model_config.device)
        return model
    
    def get_new_model(self, operation:Config.Operation):
        self.new_model_config.set_path(operation)
        new_model = self.load_model(self.new_model_config)
        return new_model

class Partition(Prototype):
    '''
    Split test set by a dv threshold or model mistakes, and then feed test splits into models. \n
    If split with threshold, then test set can get dv when setting up this checker.\n
    If split with mistakes, TBD
    '''
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)

    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        super().setup(old_model_config, datasplits, operation, plot) 
        self.test_info = DataStat.build_info(datasplits, 'test_shift', self.clf, self.new_model_config.batch_size, self.new_model_config.new_batch_size)

    def get_subset_loader(self, acquisition_bound):
        loader = Partitioner.Threshold().run(self.test_info, acquisition_bound)        
        return loader

    def run(self, operation: Config.Operation):
        new_model = self.get_new_model(operation)
        test_loader = self.get_subset_loader(operation.acquisition.bound)
        return self._target_test(test_loader, new_model)

    def _target_test(self, loader, new_model):
        '''
        loader: new_model + old_model
        '''

        if len(loader['old_model']) == 0:
            print('Nothing for Old Model')
            gt,pred,_  = new_model.eval(loader['new_model'])
            new_correct = (gt==pred)
            total_correct = new_correct

        elif len(loader['new_model']) == 0:
            print('Nothing for New Model')
            gt,pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (gt==pred)
            total_correct = old_correct

        else:
            gt,pred,_  = new_model.eval(loader['new_model'])
            new_correct = (gt==pred)
            gt,pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
        
        # DataStat.pred_metric(loader['new_model'], self.base_model, new_model)
        # print('ACC compare:', total_correct.mean()*100, self.base_acc)
        return total_correct.mean()*100 - self.base_acc 
    
    def iter_test(self):
        new_model = Model.load(self.new_model_config)
        acc_change = []
        for threshold in self.threshold_collection:
            loader = Partitioner.Threshold().run(self.test_info,threshold)
            gt,pred,_  = Model.evaluate(loader['new_model'],new_model)
            new_correct = (gt==pred)
            gt,pred,_  = Model.evaluate(loader['old_model'], self.base_model)
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
            assert total_correct.size == self.test_info['gt'].size
            acc_change.append(total_correct.mean()*100-self.base_acc)
        return acc_change

class Probability(Partition):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        super().setup(old_model_config, datasplits, operation, plot) 
        self.anchor_loader = datasplits.loader['val_shift'] # keep for Seq
        anchor_dstr = self.set_up_dstr(self.anchor_loader, operation.stream.pdf)
        self.test_loader, posteriors = self.get_subset_loader(anchor_dstr, operation.stream)
        print('Set up: ')
        # self.mistake_stat(self.test_loader['new_model'], 'New Model Test')
        if plot:
            pdf_name = self.get_pdf_name(operation.stream.pdf)
            fig_name = 'figure/test/probab.png'
            self.probab_dstr_plot(posteriors, fig_name)
            fig_name = 'figure/test/dv.png'
            self.mistake_stat(datasplits.loader['test_shift'], plot=True, plot_name=fig_name, pdf=operation.stream.pdf)
            self.selected_dstr_plot(self.test_loader['new_model'], fig_name, operation.stream.pdf)

    def set_up_dstr(self, set_up_loader, pdf_type):
        correct_dstr = Distribution.disrtibution(self.base_model, self.clf, set_up_loader, pdf_type, correctness=True)
        incorrect_dstr = Distribution.disrtibution(self.base_model, self.clf, set_up_loader, pdf_type, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}

    def get_subset_loader(self, anchor_dstr, stream_instruction:Config.ProbabStream):
        loader, posteriors = Partitioner.Probability().run(self.test_info, {'target': anchor_dstr['incorrect'], 'other': anchor_dstr['correct']}, stream_instruction)
        return loader, posteriors
    
    def selected_dstr_plot(self, selected_loader, fig_name, pdf=None):
        self.mistake_stat(selected_loader, plot=True, plot_name=fig_name, pdf=pdf, loader_name='Selected Test')

    def run(self, operation:Config.Operation):
        '''
        Use a new CLF and reset DSTR, test_loader in testing seq
        '''
        new_model = self.get_new_model(operation)
        if 'seq' in operation.acquisition.method:
            clf_log = Log(self.new_model_config, 'clf')
            self.clf = clf_log.import_log(operation)
            anchor_dstr = self.set_up_dstr(self.anchor_loader, operation.stream.pdf)
            self.test_loader, _ = self.get_subset_loader(anchor_dstr, operation.stream)
            print('Seq Running:')
            self.mistake_stat(self.test_loader['new_model'], 'New Test')
        return self._target_test(self.test_loader, new_model)

    def _target_test(self, loader, new_model):
        return super()._target_test(loader, new_model)
    
    def get_pdf_name(self, pdf_method):
        return '' if pdf_method == None else '_{}'.format(pdf_method)

    def probab_dstr_plot(self, probab, fig_name, pdf_method=None):
        test_loader = torch.utils.data.DataLoader(self.test_info['dataset'], batch_size=self.new_model_config.batch_size)
        dataset_gts, dataset_preds, _ = self.base_model.eval(test_loader)
        correct_mask = (dataset_gts == dataset_preds)
        Distribution.base_plot(probab[correct_mask], 'correct', 'green', pdf_method)        
        incorrect_mask = ~correct_mask
        Distribution.base_plot(probab[incorrect_mask], 'incorrect', 'red', pdf_method)  
        Distribution.plt.xlabel('Probability')
        Distribution.plt.savefig(fig_name)
        Distribution.plt.close()
        print('Save fig to {}'.format(fig_name))

    def dv_dstr_plot(self, cor_dv, incor_dv, fig_name, pdf_method=None):
        Distribution.base_plot(cor_dv, 'correct', 'orange', pdf_method)
        Distribution.base_plot(incor_dv, 'incorrect', 'blue', pdf_method)
        Distribution.plt.xlabel('Decision Value')
        Distribution.plt.ylabel('Density')
        Distribution.plt.savefig(fig_name)
        Distribution.plt.close()
        print('Save fig to {}'.format(fig_name))
   
    def mistake_stat(self, dataloader, loader_name=None, plot=False, plot_name=None, pdf=None):
        cor_dv = DataStat.get_correctness_dv(self.base_model, dataloader, self.clf, correctness=True)
        incor_dv = DataStat.get_correctness_dv(self.base_model, dataloader, self.clf, correctness=False)
        if loader_name is not None:
            print('Hard images in {}: {}%'.format(loader_name, len(incor_dv) / (len(incor_dv) + len(cor_dv)) * 100))
        if plot:
            self.dv_dstr_plot(cor_dv, incor_dv, plot_name, pdf)

class Ensemble(Prototype):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        super().setup(old_model_config, datasplits, operation, plot)
        self.pdf_type = operation.stream.pdf
        self.probab_partitioner = Partitioner.Probability()
        self.test_loader = datasplits.loader['test_shift']
    
    def run(self, operation:Config.Operation):
        new_model = self.get_new_model(operation)
        return self._target_test(self.test_loader, new_model)

    def ensemble_decision(self, new_probab, new_weights, old_probab, old_weights):
        '''
        Ensembled Decisions for Each Class
        '''
        return new_probab * new_weights + old_probab * old_weights
    
    @abstractmethod
    def get_weight(self):
        pass
    
class DstrEnsemble(Ensemble):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)
    
    def setup(self, old_model_config: Config.OldModel, datasplits: Dataset.DataSplits, operation: Config.Operation, plot: bool):
        super().setup(old_model_config, datasplits, operation, plot)
        self.anchor_dstr = self.set_up_dstr(datasplits.loader['val_shift'], self.pdf_type)

    def set_up_dstr(self, set_up_loader, pdf_type):
        correct_dstr = Distribution.disrtibution(self.base_model, self.clf, set_up_loader, pdf_type, correctness=True)
        incorrect_dstr = Distribution.disrtibution(self.base_model, self.clf, set_up_loader, pdf_type, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}

    def get_weight(self, dstr_dict, observations, size):
        weights = []
        for value in observations:
            posterior = self.probab_partitioner.get_posterior(value, dstr_dict, self.pdf_type)
            weights.append(posterior)
        return np.concatenate(weights).reshape((size,1))

    def _target_test(self, dataloader, new_model):
        dv, _ = self.clf.predict(dataloader)  
        size = len(dv)

        decision_maker = Decision.factory(self.new_model_config.base_type, self.new_model_config.class_number)
        new_decision_probab = decision_maker.get(new_model, dataloader)
        old_decision_probab = decision_maker.get(self.base_model, dataloader)

        new_weight = self.get_weight({'target': self.anchor_dstr['incorrect'], 'other': self.anchor_dstr['correct']}, dv, size)
        old_weight = self.get_weight({'target': self.anchor_dstr['correct'], 'other': self.anchor_dstr['incorrect']}, dv, size)

        probab = self.ensemble_decision(new_decision_probab, new_weight, old_decision_probab, old_weight)
        decision = decision_maker.apply(probab)

        gts = dataloader_utils.get_labels(dataloader)
        final_acc = (gts==decision).mean() * 100 
        print('ACC compare:',final_acc, self.base_acc)

        DataStat.pred_metric(dataloader, self.base_model, new_model)
        
        return final_acc - self.base_acc   
    
class AverageEnsemble(Ensemble):
    def __init__(self, model_config: Config.NewModel, general_config) -> None:
        super().__init__(model_config, general_config)

    def get_weight(self, size):
        return np.repeat([0.5], size).reshape((size,1))
    
    def _target_test(self, dataloader, new_model):
        gts = dataloader_utils.get_labels(dataloader)
        size = len(gts)
        new_probab = self.get_decision(new_model, dataloader)
        old_probab = self.get_decision(self.base_model, dataloader)
        new_weight = self.get_weight(size)
        old_weight = self.get_weight(size)

        probab = self.ensemble_decision(new_probab, new_weight, old_probab, old_weight)

        preds = np.argmax(probab, axis=-1)

        DataStat.pred_metric(dataloader, self.base_model, new_model)

        final_acc = (gts==preds).mean() * 100 

        print('ACC compare:',final_acc, self.base_acc)
        return final_acc - self.base_acc   
     
# class AdaBoostEnsemble(Ensemble):
#     def __init__(self, model_config: Config.NewModel) -> None:
#         super().__init__(model_config)
    
#     def get_boosting_alpha(self, model:Model.Prototype, dataloader):
#         gts, preds, _  = model.eval(dataloader)
#         err_mask = (gts!=preds)
#         total_err = err_mask.mean()
#         alpha = np.log((1 - total_err) / total_err) / 2
#         print(total_err, alpha)
#         return alpha
    
#     def ensemble_probab(self, new_probab, old_probab, new_model, old_model, anchor_dataloader):
#         new_alpha = self.get_boosting_alpha(new_model, anchor_dataloader)
#         old_alpha = self.get_boosting_alpha(old_model, anchor_dataloader)
#         probab = new_probab * new_alpha + old_probab * old_alpha
#         return probab

# class MaxDstr(DstrEnsemble):
#     def __init__(self, model_config: Config.NewModel) -> None:
#         super().__init__(model_config)

#     def ensemble_probab(self, dataloader, new_probab, old_probab):
#         old_weights, new_weights = self.get_dstr_weights(dataloader, self.pdf_type)
#         return np.max(old_weights * old_probab, new_weights * new_probab)
        
    
# class MaxAverage(AverageEnsemble):
#     def __init__(self, model_config: Config.NewModel) -> None:
#         super().__init__(model_config)
#     def ensemble_probab(self, dataloader, new_probab, old_probab):
#         return max(new_probab, old_probab)
    
def factory(name, new_model_config, general_config):
    if name == 'subset':
        checker = Partition(new_model_config, general_config)
    elif name == 'probab':
        checker = Probability(new_model_config, general_config)
    elif name == 'dstr':
        checker = DstrEnsemble(new_model_config, general_config)
    elif name == 'avg':
        checker = AverageEnsemble(new_model_config, general_config)
    # elif name == 'max_dstr':
    #     checker = MaxDstr(new_model_config)
    # elif name == 'max_avg':
    #     checker = MaxAverage(new_model_config)
    else:
        checker = Prototype(new_model_config, general_config)
    return checker

def get_configs(epoch, parse_param, dataset):
    model_dir, device_config, base_type, pure, new_model_setter, general_config = parse_param

    batch_size = general_config['hparams']['batch_size']
    superclass_num = general_config['hparams']['superclass']

    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, base_type=base_type)
    # new_model_dir = model_dir[:2] if dev_name == 'sm' else model_dir
    new_model_dir = model_dir # For imbalanced test and market filtering
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, new_model_dir, device_config, epoch, pure, new_model_setter, batch_size['new'], base_type=base_type)
    dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)
    return old_model_config, new_model_config, dataset_splits, general_config

def instantiate(epoch, parse_args, dataset, operation: Config.Operation, plot=True):
    old_model_config, new_model_config, dataset_splits, general_config = get_configs(epoch, parse_args, dataset)
    checker = factory(operation.stream.name, new_model_config, general_config)
    checker.setup(old_model_config, dataset_splits, operation, plot)
    return checker

class total(Prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        self.base_model = Model.load(old_model_config)
        self.test_loader = datasplits.loader['test_shift']


    def run(self, acquisition_config, recall=False):
        base_gt, base_pred, _ = Model.evaluate(self.test_loader,self.base_model)
        self.new_model_config.set_path(acquisition_config)
        new_model = Model.load(self.new_model_config)
        new_gt, new_pred, _ = Model.evaluate(self.test_loader,new_model)
        if recall:
            base_cls_mask = (base_gt==0)
            base_recall = ((base_gt==base_pred)[base_cls_mask]).mean()*100
            cls_mask = (new_gt==0)
            new_recall = ((new_gt==new_pred)[cls_mask]).mean()*100
            return new_recall - base_recall
        else:
            base_acc = (base_gt==base_pred).mean()*100
            acc = (new_gt==new_pred).mean()*100
            return acc - base_acc