import binary.utils.objects.Config as Config
import binary.utils.objects.dataset as Dataset
import utils.objects.Config as Config
import utils.objects.Detector as Detector
import utils.objects.dataset as Dataset
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.Detector as Detector
import utils.objects.dataset as Dataset
import utils.log as log
import utils.statistics.stat_test as Plot_Stat
from abc import abstractmethod
import numpy as np
import torch
import utils.statistics.subset as Subset
class prototype():
    def __init__(self,model_config:Config.NewModel) -> None:
        self.model_config = model_config
    @abstractmethod
    def run(self,acquisition_config:Config.Acquistion):
        pass
    @abstractmethod
    def setup(self, old_model_config:Config.OldModel, data_splits):
        '''
        Use the old model and data split to set up a prototype (for each epoch)
        '''
        pass
class DV(prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
        # prototype.__init__(self, model_config) # TODO: a problem in the constructor of multi-inheritence MRO
        self.log_config = log.get_config(model_config)
    def load(self):
        data = torch.load(self.log_config.path)
        print('load DV from {}'.format(self.log_config.path))
        return data
    def setup(self, old_model_config, datasplits):
        self.base_model = Model.load(old_model_config)
        clf,clip_processor,_ = Detector.get_CLF(self.base_model,datasplits.loader)
        self.clf = clf
        self.clip_processor = clip_processor   
    def run(self,acquisition_config):
        self.log_config.set_path(acquisition_config)
        data = self.log.load()
        loader = torch.utils.data.DataLoader(data, batch_size=self.model_config.batch_size, shuffle=True,drop_last=True)
        data_info, _ = Detector.apply_CLF(self.clf, loader, self.clip_processor, self.base_model)
        return np.round(np.mean(data_info['dv']),decimals=3)
class benchmark(DV):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        super().setup(old_model_config, datasplits)
        datasplits = datasplits       
    def run(self, datasplits):
        data_info, _ = Detector.apply_CLF(self.clf, datasplits.loader['market'], self.clip_processor, self.base_model)
        # return (data_info['dv']<0).sum()/len(data_info['dv'])     
        return np.std(data_info['dv']), np.mean(data_info['dv'])
class subset(prototype):
    '''
    Split test set by a dv threshold or model mistakes, and then feed test splits into models. \n
    If split with threshold, then test set can get dv when setting up this checker.\n
    If split with mistakes, TBD
    '''
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    
    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, detector_instruction:Config.Dectector):
        self.base_model = Model.prototype_factory(old_model_config.base_type, old_model_config.class_number, detector_instruction.vit)
        self.base_model.load(old_model_config.path, old_model_config.device)
        self.clf = Detector.factory(detector_instruction.name, clip_processor = detector_instruction.vit, split_and_search=True)
        _ = self.clf.fit(self.base_model, datasplits.loader['val_shift'])
        gt, pred, _ = self.base_model.eval(datasplits.loader['test_shift'])
        self.base_acc = (gt == pred).mean()*100 
        self.test_info = Subset.build_data_info(datasplits, 'test_shift', self.clf, self.model_config, self.base_model)
        self.vit = detector_instruction.vit

    def get_subset_loader(self, threshold):
        loader = Subset.threshold_subset_setter().get_subset_loders(self.test_info,threshold)        
        return loader

    def run(self, acquisition_config:Config.Acquistion, stream_instruction: Config.DVStream):
        loader = self.get_subset_loader(stream_instruction.bound)
        self.model_config.set_path(acquisition_config)
        return self._target_test(loader)

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
        
        Subset.pred_metric(loader['new_model'], self.base_model, new_model)
        return total_correct.mean()*100 - self.base_acc 
    
    def iter_test(self):
        new_model = Model.load(self.model_config)
        acc_change = []
        for threshold in self.threshold_collection:
            loader = Subset.threshold_subset_setter().get_subset_loders(self.test_info,threshold)
            gt,pred,_  = Model.evaluate(loader['new_model'],new_model)
            new_correct = (gt==pred)
            gt,pred,_  = Model.evaluate(loader['old_model'], self.base_model)
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
            assert total_correct.size == self.test_info['gt'].size
            acc_change.append(total_correct.mean()*100-self.base_acc)
        return acc_change

class probability(subset):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)  

    def get_dv_dstr(self, dataloader, pdf_type):
        '''
        Get decision value distribution of a dataloader under a base model
        '''
        cor_dv, incor_dv = Plot_Stat.get_dv_dstr(self.base_model, dataloader, self.clf)
        # total_dv = np.concatenate((incor_dv,cor_dv))
        # print('Negative DV: {}%'.format((total_dv<0).mean()*100))
        correct_prior = (len(cor_dv)) / (len(cor_dv) + len(incor_dv))
        correct_dstr = Subset.disrtibution(correct_prior, Plot_Stat.get_pdf(cor_dv, pdf_type))
        incorrect_dstr =  Subset.disrtibution(1 - correct_prior, Plot_Stat.get_pdf(incor_dv, pdf_type))
        return correct_dstr, incorrect_dstr

    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, detector_instruction: Config.Dectector, stream_instruction:Config.ProbabStream, plot:bool):
        super().setup(old_model_config, datasplits, detector_instruction) 
        correct_dstr, incorrect_dstr = self.get_dv_dstr(datasplits.loader['val_shift'], stream_instruction.pdf)
        self.loader, selected_probab = Subset.probability_setter().get_subset_loders(self.test_info, correct_dstr, incorrect_dstr, stream_instruction)
        cor_dv, incor_dv = Plot_Stat.get_dv_dstr(self.base_model, self.loader['new_model'], self.clf)
        print('Hard images in New Test: {}%'.format(len(incor_dv) / (len(incor_dv) + len(cor_dv)) * 100))
        if plot:
            pdf_name = self.get_pdf_name(stream_instruction.pdf)
            fig_name = 'figure/test/probab.png'
            self.probab_dstr_plot(selected_probab, fig_name)
            fig_name = 'figure/test/dv.png'
            cor_dv, incor_dv = Plot_Stat.get_dv_dstr(self.base_model, datasplits.loader['test_shift'], self.clf)
            self.dv_dstr_plot(cor_dv, incor_dv, fig_name, stream_instruction.pdf)
            fig_name = 'figure/test/selected_{}{}.png'.format(stream_instruction.bound, pdf_name)
            self.selected_dstr_plot(self.loader['new_model'], fig_name, stream_instruction.pdf)

    def selected_dstr_plot(self, selected_loader, fig_name, pdf=None):
        cor_dv, incor_dv = Plot_Stat.get_dv_dstr(self.base_model, selected_loader, self.clf)
        print('Old model mistakes in selected test: {}%'.format(len(incor_dv) / (len(incor_dv) + len(cor_dv)) * 100))
        self.dv_dstr_plot(cor_dv, incor_dv, fig_name, pdf)

    def run(self, acquisition_config:Config.Acquistion):
        self.model_config.set_path(acquisition_config)
        return self._target_test(self.loader)

    def _target_test(self, loader):
        return super()._target_test(loader)
    
    def get_pdf_name(self, pdf_method):
        return '' if pdf_method == None else '_{}'.format(pdf_method)

    def probab_dstr_plot(self, probab, fig_name, pdf_method=None):
        test_loader = torch.utils.data.DataLoader(self.test_info['dataset'], batch_size=self.model_config.batch_size)
        dataset_gts, dataset_preds, _ = self.base_model.eval(test_loader)
        correct_mask = (dataset_gts == dataset_preds)
        Plot_Stat.base_plot(probab[correct_mask], 'correct', 'green', pdf_method)        
        incorrect_mask = ~correct_mask
        Plot_Stat.base_plot(probab[incorrect_mask], 'incorrect', 'red', pdf_method)  
        Plot_Stat.plt.xlabel('Probability')
        Plot_Stat.plt.savefig(fig_name)
        Plot_Stat.plt.close()
        print('Save fig to {}'.format(fig_name))

    def dv_dstr_plot(self, cor_dv, incor_dv, fig_name, pdf_method=None):
        Plot_Stat.base_plot(cor_dv, 'correct', 'orange', pdf_method)
        Plot_Stat.base_plot(incor_dv, 'incorrect', 'blue', pdf_method)
        # Plot_Stat.plt.savefig()
        Plot_Stat.plt.xlabel('Decision Value')
        Plot_Stat.plt.ylabel('Density')
        Plot_Stat.plt.savefig(fig_name)
        Plot_Stat.plt.close()
        print('Save fig to {}'.format(fig_name))

class ensemble(probability):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config: Config.OldModel, datasplits: Dataset.DataSplits, detector_instruction: Config.Dectector, stream_instruction: Config.ProbabStream, plot: bool):
        self.base_model = Model.prototype_factory(old_model_config.base_type, old_model_config.class_number, detector_instruction.vit)
        self.base_model.load(old_model_config.path, old_model_config.device)
        self.clf = Detector.factory(detector_instruction.name, clip_processor = detector_instruction.vit, split_and_search=True)
        _ = self.clf.fit(self.base_model, datasplits.loader['val_shift'])
        gt, pred, _ = self.base_model.eval(datasplits.loader['test_shift'])
        self.base_acc = (gt == pred).mean() * 100 
        self.pdf_type = stream_instruction.pdf
        self.loader = datasplits.loader['test_shift']
        self.probab_setter = Subset.probability_setter()
        self.vit = detector_instruction.vit
        self.anchor_loader = datasplits.loader['val_shift']

    def get_dstr_weights(self, dataloader, pdf_type):
        correct_dstr, incorrect_dstr = self.get_dv_dstr(dataloader, pdf_type)
        old_weights, new_weights = [], []
        dv, _ = self.clf.predict(dataloader, self.base_model)        
        for value in dv:
            new_probab = self.probab_setter.get_probab(value, incorrect_dstr, correct_dstr, self.pdf_type)
            new_weights.append(new_probab)
            old_probab = self.probab_setter.get_probab(value, correct_dstr, incorrect_dstr, self.pdf_type)
            old_weights.append(old_probab)
            # assert old_probab[0] + new_probab[0] == 1, old_probab[0] + new_probab[0] 
        return np.concatenate(old_weights), np.concatenate(new_weights)
    
    def _target_test(self, dataloader):
        new_model = Model.prototype_factory(self.model_config.base_type, self.model_config.class_number, self.vit)
        new_model.load(self.model_config.path, self.model_config.device)     
        gts, _, new_probab  = new_model.eval(dataloader)
        _, _, old_probab  = self.base_model.eval(dataloader)  
        preds = np.zeros(len(gts))
        probab = self.average(new_probab, old_probab)
        # probab = self.weight_error(new_probab, old_probab, new_model, self.base_model, self.anchor_loader)
        if self.model_config.class_number == 1:
            preds[probab >= 0.5] = 1
        else:
            preds = np.argmax(probab, axis=0)
        return (gts==preds).mean() * 100 - self.base_acc     

    def weight_dstr(self, dataloader, new_probab, old_probab):
        old_weights, new_weights = self.get_dstr_weights(dataloader, self.pdf_type)
        probab = new_probab * new_weights + old_probab * old_weights
        return probab
    
    def average(self, new_probab, old_probab):
        probab = (new_probab + old_probab) / 2
        return probab

    def get_boosting_alpha(self, model:Model.prototype, dataloader):
        gts, preds, _  = model.eval(dataloader)
        err_mask = (gts!=preds)
        total_err = err_mask.mean()
        alpha = np.log((1 - total_err) / total_err) / 2
        # print(total_err, alpha)
        return alpha
    
    def weight_error(self, new_probab, old_probab, new_model, old_model, anchor_dataloader):
        new_alpha = self.get_boosting_alpha(new_model, anchor_dataloader)
        old_alpha = self.get_boosting_alpha(old_model, anchor_dataloader)
        probab = new_probab * new_alpha + old_probab * old_alpha
        return probab

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