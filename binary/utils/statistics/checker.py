import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.Detector as Detector
import utils.objects.dataset as Dataset
import utils.log as log
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
    def __init__(self, model_config: Config.NewModel, clip_processor: Detector.CLIPProcessor, clip_set_up_loader) -> None:
        super().__init__(model_config)
        self.clip_processor = clip_processor
        self.clip_set_up_loader = clip_set_up_loader

    def setup(self, old_model_config:Config.OldModel, datasplits:Dataset.DataSplits, new_model_config: Config.NewModel):
        self.base_model = Model.prototype_factory(old_model_config.base_type, self.clip_set_up_loader, self.clip_processor)
        self.base_model.load(old_model_config)
        # TODO: CLF can be made temporary
        self.clf = Detector.SVM(self.clip_set_up_loader, self.clip_processor)
        _ = self.clf.fit(self.base_model, datasplits.loader['val_shift'])
        gt, pred, _ = self.base_model.eval(datasplits.loader['test_shift'])
        self.base_acc = (gt == pred).mean()*100 
        self.test_info = self.get_test_info(datasplits, new_model_config)  

    def get_test_info(self, datasplits:Dataset.DataSplits, new_model_config: Config.NewModel):
        test_info = {}
        test_dv, _ = self.clf.predict(datasplits.loader['test_shift'])        
        test_info['dv'] = test_dv
        test_info['old_batch_size'] = new_model_config.batch_size
        test_info['new_batch_size'] = new_model_config.new_batch_size
        test_info['dataset'] = datasplits.dataset['test_shift']
        test_info['loader'] = datasplits.loader['test_shift']
        return test_info

    def get_subset_loader(self, threshold):
        loader = Subset.threshold_subset_setter().get_subset_loders(self.test_info,threshold)        
        return loader
    
    # def set_subset_loader(self, loader):
    #     self.test_info['loader'] = loader

    def run(self, threshold, acquisition_config):
        loader = self.get_subset_loader(threshold)
        self.model_config.set_path(acquisition_config)
        return self._target_test(loader)

    def _target_test(self, loader):
        '''
        loader: new_model + old_model
        '''
        new_model = Model.prototype_factory(self.model_config.base_type, self.clip_set_up_loader, self.clip_processor)
        new_model.load(self.model_config)
        gt,pred,_  = new_model.eval(loader['new_model'])
        new_correct = (gt==pred)
        if len(loader['old_model']) == 0:
            total_correct = new_correct
        else:
            gt,pred,_  = self.base_model.eval(loader['old_model'])
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
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

def factory(check_method, model_config,  clip_processor: Detector.CLIPProcessor, clip_set_up_loader):
    if check_method == 'dv':
        product = DV(model_config)
    elif check_method == 'total':
        product = total(model_config)
    elif check_method == 'bm':
        product = benchmark(model_config)
    else:
        product = subset(model_config, clip_processor, clip_set_up_loader) 
    return product
