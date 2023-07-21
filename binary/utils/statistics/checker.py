import binary.utils.objects.Config as Config
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

class prototype():
    '''
    Decide what to test the new and the old model
    '''
    def __init__(self,model_config:Config.NewModel) -> None:
        self.model_config = model_config
    @abstractmethod
    def run(self, operation:Config.Operation):
        '''
        Load the new model and perform testing
        '''
        pass
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        '''
        Use the old model and data split to set up a prototype (for each epoch) -> base_model, clf/detector, anchor loader
        '''
        self.base_model = Model.prototype_factory(old_model_config.base_type, old_model_config.class_number, operation.detection.vit)
        self.base_model.load(old_model_config.path, old_model_config.device)
        self.clf = Detector.factory(operation.detection.name, clip_processor = operation.detection.vit, split_and_search=True)
        _ = self.clf.fit(self.base_model, datasplits.loader['val_shift'])
        self.anchor_loader = datasplits.loader['val_shift']

class Partition(prototype):
    '''
    Split test set by a dv threshold or model mistakes, and then feed test splits into models. \n
    If split with threshold, then test set can get dv when setting up this checker.\n
    If split with mistakes, TBD
    '''
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        '''
        set base model, test data info and vit
        '''
        super().setup(old_model_config, datasplits, operation, plot) 
        self.base_acc = self.base_model.acc(datasplits.loader['test_shift'])
        self.test_info = DataStat.build_info(datasplits, 'test_shift', self.clf, self.model_config.batch_size, self.model_config.new_batch_size, self.base_model)
        self.vit = operation.detection.vit

    def get_subset_loader(self, threshold):
        loader = Partitioner.threshold_subset_setter().get_subset_loders(self.test_info,threshold)        
        return loader

    def run(self, operation:Config.Operation):
        self.test_loader = self.get_subset_loader(operation.acquisition.bound)
        self.model_config.set_path(operation)
        return self._target_test(self.test_loader)

    def _target_test(self, loader):
        '''
        loader: new_model + old_model
        '''
        new_model = Model.prototype_factory(self.model_config.base_type, self.model_config.class_number, self.vit)
        new_model.load(self.model_config.path, self.model_config.device)
        gt,pred,_  = new_model.eval(loader['new_model'])
        new_correct = (gt==pred)

        if len(loader['old_model']) == 0:
            total_correct = new_correct
        else:
            gt,pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
        
        DataStat.pred_metric(loader['new_model'], self.base_model, new_model)
        return total_correct.mean()*100 - self.base_acc 
    
    def iter_test(self):
        new_model = Model.load(self.model_config)
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
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)  
   
    def mistake_stat(self, dataloader, loader_name=None, plot=False, plot_name=None, pdf=None):
        cor_dv = DataStat.get_correctness_dv(self.base_model, dataloader, self.clf, correctness=True)
        incor_dv = DataStat.get_correctness_dv(self.base_model, dataloader, self.clf, correctness=False)
        if loader_name is not None:
            print('Hard images in {}: {}%'.format(loader_name, len(incor_dv) / (len(incor_dv) + len(cor_dv)) * 100))
        if plot:
            self.dv_dstr_plot(cor_dv, incor_dv, plot_name, pdf)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        super().setup(old_model_config, datasplits, operation, plot) 
        self.test_loader, selected_probab = self.get_subset_loader(operation.stream, self.anchor_loader)
        print('Set up: ')
        self.mistake_stat(self.test_loader['new_model'], 'New Model Test')
        if plot:
            pdf_name = self.get_pdf_name(operation.stream.pdf)
            fig_name = 'figure/test/probab.png'
            self.probab_dstr_plot(selected_probab, fig_name)
            fig_name = 'figure/test/dv.png'
            self.mistake_stat(datasplits.loader['test_shift'], plot=True, plot_name=fig_name, pdf=operation.stream.pdf)
            self.selected_dstr_plot(self.test_loader['new_model'], fig_name, operation.stream.pdf)

    def get_subset_loader(self, stream_instruction:Config.Stream, set_up_loader):
        correct_dstr = Distribution.get_correctness_dstr(self.base_model, self.clf, set_up_loader, stream_instruction.pdf, correctness=True)
        incorrect_dstr = Distribution.get_correctness_dstr(self.base_model, self.clf, set_up_loader, stream_instruction.pdf, correctness=False)
        loader, selected_probab = Partitioner.Probability().run(self.test_info, {'target': incorrect_dstr, 'other': correct_dstr}, stream_instruction)
        return loader, selected_probab
    
    def selected_dstr_plot(self, selected_loader, fig_name, pdf=None):
        self.mistake_stat(selected_loader, plot=True, plot_name=fig_name, pdf=pdf, loader_name='Selected Test')

    def run(self, operation:Config.Operation):
        '''
        Use a new CLF (new dv dstr) in testing seq
        '''
        self.model_config.set_path(operation)
        if 'seq' in operation.acquisition.method:
            log = Log(self.model_config, 'clf')
            self.clf = log.import_log(operation)
            self.test_loader, _ = self.get_subset_loader(operation.stream, self.anchor_loader)
            print('Seq Running:')
            self.mistake_stat(self.test_loader['new_model'], 'New Test')
        return self._target_test(self.test_loader)

    def _target_test(self, loader):
        return super()._target_test(loader)
    
    def get_pdf_name(self, pdf_method):
        return '' if pdf_method == None else '_{}'.format(pdf_method)

    def probab_dstr_plot(self, probab, fig_name, pdf_method=None):
        test_loader = torch.utils.data.DataLoader(self.test_info['dataset'], batch_size=self.model_config.batch_size)
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

class Ensemble(Partition):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, operation:Config.Operation, plot:bool):
        super().setup(old_model_config, datasplits, operation, plot)
        self.pdf_type = operation.stream.pdf
        self.probab_partitioner = Partitioner.Probability()
        self.test_loader = torch.utils.data.DataLoader(self.test_info['dataset'], batch_size=self.model_config.new_batch_size)
    
    def run(self, operation:Config.Operation):
        self.model_config.set_path(operation)
        return self._target_test(self.test_loader)
    
    def get_decision(self, model:Model.prototype, dataloader):
        _, _, decision  = model.eval(dataloader)
        if self.model_config.class_number == 1:
            decision = self.transform_BDE_probab(decision)
        return decision

    def ensemble_decision(self, new_probab, new_weights, old_probab, old_weights):
        '''
        Ensembled Decisions for Each Class
        '''
        return new_probab * new_weights + old_probab * old_weights
    
    def transform_BDE_probab(self, class_1_probab):
        '''
            Make 1D sigmoid output 2D
        '''
        size = len(class_1_probab)
        class_0_probab = (1-class_1_probab).reshape((size,1))
        class_1_probab = class_1_probab.reshape((size,1))
        return np.concatenate((class_0_probab, class_1_probab), axis=1)
    
    @abstractmethod
    def get_weight(self):
        pass

    def get_gts(self, dataloader):
        gts = []
        for batch_info in dataloader:
            gts.append(batch_info[1].cpu())
        return torch.concat(gts).numpy()
    
class AverageEnsemble(Ensemble):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)

    def get_weight(self, size):
        return np.repeat([0.5], size).reshape((size,1))
    
    def _target_test(self, dataloader):
        gts = self.get_gts(dataloader)
        size = len(gts)
        new_model = Model.prototype_factory(self.model_config.base_type, self.model_config.class_number, self.vit)
        new_model.load(self.model_config.path, self.model_config.device)  
        new_probab = self.get_decision(new_model, dataloader)
        old_probab = self.get_decision(self.base_model, dataloader)
        new_weight = self.get_weight(size)
        old_weight = self.get_weight(size)

        probab = self.ensemble_decision(new_probab, new_weight, old_probab, old_weight)

        preds = np.argmax(probab, axis=-1)

        DataStat.pred_metric(dataloader, self.base_model, new_model)
        return (gts==preds).mean() * 100 - self.base_acc 
     
class DstrEnsemble(Ensemble):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    
    def get_correctness_dstr(self, dataloader, pdf_type, correctness):
        dstr = Distribution.get_correctness_dstr(self.base_model, self.clf, dataloader, pdf_type, correctness=correctness)
        return dstr

    def get_weight(self, dstr_dict, observations, size):
        weights = []
        for value in observations:
            posterior = self.probab_partitioner.get_posterior(value, dstr_dict, self.pdf_type)
            weights.append(posterior)
        return np.concatenate(weights).reshape((size,1))

    def _target_test(self, dataloader):
        gts = self.get_gts(dataloader)
        size = len(gts)
        dv, _ = self.clf.predict(dataloader, self.base_model)  

        new_model = Model.prototype_factory(self.model_config.base_type, self.model_config.class_number, self.vit)
        new_model.load(self.model_config.path, self.model_config.device)  
        new_probab = self.get_decision(new_model, dataloader)
        old_probab = self.get_decision(self.base_model, dataloader)
        correct_dstr = self.get_correctness_dstr(dataloader, self.pdf_type, correctness=True)
        incorrect_dstr = self.get_correctness_dstr(dataloader, self.pdf_type, correctness=False)
        new_weight = self.get_weight({'target': incorrect_dstr, 'other': correct_dstr}, dv, size)
        old_weight = self.get_weight({'target': correct_dstr, 'other': incorrect_dstr}, dv, size)

        probab = self.ensemble_decision(new_probab, new_weight, old_probab, old_weight)
        preds = np.argmax(probab, axis=-1)

        DataStat.pred_metric(dataloader, self.base_model, new_model)
        return (gts==preds).mean() * 100 - self.base_acc   
    
# class AdaBoostEnsemble(Ensemble):
#     def __init__(self, model_config: Config.NewModel) -> None:
#         super().__init__(model_config)
    
#     def get_boosting_alpha(self, model:Model.prototype, dataloader):
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
    
def factory(name, new_model_config):
    if name == 'subset':
        checker = Partition(new_model_config)
    elif name == 'probab':
        checker = Probability(new_model_config)
    elif name == 'dstr':
        checker = DstrEnsemble(new_model_config)
    elif name == 'avg':
        checker = AverageEnsemble(new_model_config)
    # elif name == 'max_dstr':
    #     checker = MaxDstr(new_model_config)
    # elif name == 'max_avg':
    #     checker = MaxAverage(new_model_config)
    else:
        checker = prototype(new_model_config)
    return checker

def get_args(epoch, parse_args, dataset):
    batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config = parse_args
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, base_type=base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, pure, new_model_setter, batch_size['new'], base_type=base_type)
    dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)
    return old_model_config, new_model_config, dataset_splits

def instantiate(epoch, parse_args, dataset, operation: Config.Operation, plot=True):
    old_model_config, new_model_config, dataset_splits = get_args(epoch, parse_args, dataset)
    checker = factory(operation.stream.name, new_model_config)
    checker.setup(old_model_config, dataset_splits, operation, plot)
    return checker

class total(prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        self.base_model = Model.load(old_model_config)
        self.test_loader = datasplits.loader['test_shift']


    def run(self, acquisition_config, recall=False):
        base_gt, base_pred, _ = Model.evaluate(self.test_loader,self.base_model)
        self.model_config.set_path(acquisition_config)
        new_model = Model.load(self.model_config)
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