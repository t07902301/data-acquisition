import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.CLF as CLF
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
        clf,clip_processor,_ = CLF.get_CLF(self.base_model,datasplits.loader)
        self.clf = clf
        self.clip_processor = clip_processor   
    def run(self,acquisition_config):
        self.log_config.set_path(acquisition_config)
        data = self.log.load()
        loader = torch.utils.data.DataLoader(data, batch_size=self.model_config.batch_size, shuffle=True,drop_last=True)
        data_info, _ = CLF.apply_CLF(self.clf, loader, self.clip_processor, self.base_model)
        return np.round(np.mean(data_info['dv']),decimals=3)
class benchmark(DV):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        super().setup(old_model_config, datasplits)
        self.datasplits = datasplits       
    def run(self):
        data_info, _ = CLF.apply_CLF(self.clf, self.datasplits.loader['market'], self.clip_processor, self.base_model)
        # return (data_info['dv']<0).sum()/len(data_info['dv'])     
        return np.std(data_info['dv']), np.mean(data_info['dv'])
class subset(prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        self.base_model = Model.load(old_model_config)
        # state = np.random.get_state()
        clf,clip_processor,_ = CLF.get_CLF(self.base_model,datasplits.loader)
        test_info, _ = CLF.apply_CLF(clf,datasplits.loader['test'],clip_processor)
        test_info['batch_size'] = old_model_config.batch_size
        test_info['dataset'] = datasplits.dataset['test']
        test_info['loader'] = datasplits.loader['test']
        self.test_info = test_info
        gt, pred, _ = Model.evaluate(datasplits.loader['test'], self.base_model)
        self.base_acc = (gt == pred).mean()*100   
        # np.random.set_state(state) 
        self.clf = clf
        self.clip_processor = clip_processor

    def get_subset_loader(self, threshold, acquistion_config):
        self.model_config.set_path(acquistion_config=acquistion_config)
        loader = Subset.threshold_subset_setter().get_subset_loders(self.test_info,threshold)        
        return loader
    
    def set_subset_loader(self, loader):
        self.test_info['loader'] = loader

    def run(self, acquistion_config:Config.Acquistion, threshold):
        loader = self.get_subset_loader(threshold, acquistion_config)
        self.set_subset_loader(loader)
        return self._target_test(self.test_info['loader'])

    def _target_test(self, loader):
        new_model = Model.load(self.model_config)
        gt,pred,_  = Model.evaluate(loader['new_model'],new_model)
        new_correct = (gt==pred)
        if len(loader['old_model']) == 0:
            total_correct = new_correct
        else:
            gt,pred,_  = Model.evaluate(loader['old_model'], self.base_model)
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
        self.test_loader = datasplits.loader['test']
        gt, pred, _ = Model.evaluate(self.test_loader,self.base_model)
        self.base_acc = (gt==pred).mean()*100

    def run(self, acquisition_config):
        self.model_config.set_path(acquisition_config)
        new_model = Model.load(self.model_config)
        gt, pred, _ = Model.evaluate(self.test_loader,new_model)
        acc = (gt==pred).mean()*100
        return acc - self.base_acc

def factory(check_method, model_config):
    if check_method == 'dv':
        product = DV(model_config)
    elif check_method == 'total':
        product = total(model_config)
    elif check_method == 'bm':
        product = benchmark(model_config)
    else:
        product = subset(model_config) 
    return product
