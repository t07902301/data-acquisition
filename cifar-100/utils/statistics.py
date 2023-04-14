import matplotlib.pyplot as plt
from utils import config
# import utils.dataset as dataset
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.CLF as CLF
import utils.log as log
from abc import abstractmethod
import numpy as np
import os
import torch
class test_subet_setter():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_subset_loders(self, data_info):
        '''
        data_info: dict of data, gt and pred labels, batch_size, and dv(if needed)
        '''
        pass
class threshold_test_subet_setter(test_subet_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self, data_info, threshold):
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices = acquistion.extract_class_indices(c, data_info['gt'])
            cls_dv = data_info['dv'][cls_indices]
            dv_selected_indices = (cls_dv<=threshold[c])
            selected_indices_total.append(dv_selected_indices)
        selected_indices_total = np.concatenate(selected_indices_total)
        test_selected = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[selected_indices_total])
        remained_test = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[~selected_indices_total])
        test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        test_loader = {
            'new_model':test_selected_loader,
            'old_model': remained_test_loader
        }   
        print('selected test images:', len(test_selected))
        return test_loader
class misclassification_test_subet_setter(test_subet_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self,data_info):
        incorr_cls_indices = (data_info['gt'] != data_info['pred'])
        corr_cls_indices = (data_info['gt'] == data_info['pred'])
        incorr_cls_set = torch.utils.data.Subset( data_info['dataset'],np.arange(len(data_info['dataset']))[incorr_cls_indices])
        corr_cls_set = torch.utils.data.Subset( data_info['dataset'],np.arange(len(data_info['dataset']))[corr_cls_indices])
        corr_cls_loader = torch.utils.data.DataLoader(corr_cls_set, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        incorr_cls_loader = torch.utils.data.DataLoader(incorr_cls_set, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
        test_loader = {
            'new_model':incorr_cls_loader,
            'old_model': corr_cls_loader
        }   
        subset_loader = [test_loader]
        return subset_loader
    
class plotter():
    def __init__(self, select_method, plot_methods, plot_data_numbers, model_config:Config.NewModel) -> None:
        self.select_method = select_method
        self.plot_methods = plot_methods
        self.plot_data_numbers = plot_data_numbers
        pure_name = 'pure' if model_config.pure else ''
        fig_root = 'figure/{}'.format(model_config.model_dir)
        if os.path.exists(fig_root) is False:
            os.makedirs(fig_root)
        aug_name = '' if model_config.augment else '-na'
        self.fig_name = os.path.join(fig_root, '{}-{}{}.png'.format(pure_name,self.select_method, aug_name))

    def threshold_plot(self, threshold_list, acc_list):
        fig, axs = plt.subplots(2,2, sharey=True, sharex=True)
        axs = axs.flatten()
        for threshold_idx,threshold in enumerate(threshold_list):
            threshold_result = acc_list[:,:,:,threshold_idx]#(e,m,img_num,threshold) e:epochs,m:methods
            print('In threshold:',threshold)
            for m_idx,method in enumerate(self.plot_methods): 
                method_result = threshold_result[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
                method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
                if threshold_idx == 0 :
                    axs[threshold_idx].plot(self.plot_data_numbers,method_avg,label=method)
                else:
                    axs[threshold_idx].plot(self.plot_data_numbers,method_avg)
                # axs[threshold_idx].plot(img_per_cls_list,method_avg,label=method)

            axs[threshold_idx].set_xticks(self.plot_data_numbers,self.plot_data_numbers)
            if threshold_idx == 2:
                axs[threshold_idx].set_xlabel('#new images for each superclass')
                axs[threshold_idx].set_ylabel('#model accuracy change')
            axs[threshold_idx].set_title('Threshold:{}'.format(threshold))
        fig.legend(loc=2,fontsize='x-small')
        fig.tight_layout()
        fig.savefig(self.fig_name)
        fig.clf()

    def plot(self, value_list):
        if self.select_method == 'dv':
            ylabel = 'decision value'
        elif (self.select_method == 'total') or (self.select_method == 'threshold'):
            ylabel = 'model accuracy change'
        else:
            ylabel = 'market average dv'
        for m_idx,method in enumerate(self.plot_methods): 
            method_result = value_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
            method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
            plt.plot(self.plot_data_numbers,method_avg,label=method)
        plt.xticks(self.plot_data_numbers,self.plot_data_numbers)
        plt.xlabel('#new images for each superclass')
        plt.ylabel(ylabel)
        plt.legend(fontsize='small')
        plt.savefig(self.fig_name)
        plt.clf()    

    def plot_data(self,acc_list,threshold_list=None):
        if isinstance(acc_list,list):
            acc_list = np.array(acc_list)
        self.plot(acc_list)
        print('save to', self.fig_name)        

class checker():
    def __init__(self,model_config:Config.NewModel) -> None:
        self.model_config = model_config
    @abstractmethod
    def run(self,acquisition_config:Config.Acquistion):
        pass
    @abstractmethod
    def setup(self, old_model_config:Config.OldModel, data_splits):
        '''
        Use the old model and data split to set up a checker (for each epoch)
        '''
        pass
class dv_checker(checker):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
        # checker.__init__(self, model_config) # TODO: a problem in the constructor of multi-inheritence MRO
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
        data_info = CLF.apply_CLF(self.clf, loader, self.clip_processor, self.base_model)
        return np.round(np.mean(data_info['dv']),decimals=3)
class benchmark_checker(dv_checker):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        super().setup(old_model_config, datasplits)
        self.datasplits = datasplits       
    def run(self):
        data_info = CLF.apply_CLF(self.clf, self.datasplits.loader['market'], self.clip_processor, self.base_model)
        # return (data_info['dv']<0).sum()/len(data_info['dv'])     
        return np.std(data_info['dv']), np.mean(data_info['dv'])
class test_subset_checker(checker):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        self.base_model = Model.load(old_model_config)
        # state = np.random.get_state()
        clf,clip_processor,_ = CLF.get_CLF(self.base_model,datasplits.loader)
        test_info = CLF.apply_CLF(clf,datasplits.loader['test'],clip_processor)
        test_info['batch_size'] = old_model_config.batch_size
        test_info['dataset'] = datasplits.dataset['test']
        test_info['loader'] = datasplits.loader['test']
        self.test_info = test_info
        gt, pred, _ = Model.evaluate(datasplits.loader['test'], self.base_model)
        self.base_acc = (gt == pred).mean()*100   
        # np.random.set_state(state) 
        self.clf = clf
        self.clip_processor = clip_processor
    def set_test_loader(self,loader):
        self.test_info['loader'] = loader
    def run(self, acquistion_config:Config.Acquistion, threshold):
        self.model_config.set_path(acquistion_config=acquistion_config)
        loader = threshold_test_subet_setter().get_subset_loders(self.test_info,threshold)
        self.set_test_loader(loader)
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
            loader = threshold_test_subet_setter().get_subset_loders(self.test_info,threshold)
            gt,pred,_  = Model.evaluate(loader['new_model'],new_model)
            new_correct = (gt==pred)
            gt,pred,_  = Model.evaluate(loader['old_model'], self.base_model)
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
            assert total_correct.size == self.test_info['gt'].size
            acc_change.append(total_correct.mean()*100-self.base_acc)
            # gt,pred,_  = Model.evaluate(loader['new_model'], self.base_model)
            # old_correct = (gt==pred)        
            # acc_change.append(new_correct.sum()-old_correct.sum())
        return acc_change
    
class test_set_checker(checker):
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

def checker_factory(check_method, model_config):
    if check_method == 'dv':
        checker_product = dv_checker(model_config)
    elif check_method == 'total':
        checker_product = test_set_checker(model_config)
    elif check_method == 'bm':
        checker_product = benchmark_checker(model_config)
    else:
        checker_product = test_subset_checker(model_config) 
    return checker_product

def get_threshold(clf, clip_processor, acquisition_config:Config.Acquistion, model_config:Config.NewModel, market_ds):
    '''
    Use indices log and SVM to determine max decision values for each class.\n
    old_model + data -> SVM \n
    model_config + acquisition -> indices log
    '''
    if acquisition_config.method == 'seq_clf':
        return seq_bound(clf, clip_processor, acquisition_config, model_config)
    else:
        market_loader = torch.utils.data.DataLoader(market_ds, batch_size=model_config.batch_size, 
                                    num_workers=config['num_workers'])
        market_info = CLF.apply_CLF(clf, market_loader, clip_processor)
        return non_seq_bound(acquisition_config, model_config, market_info)

def non_seq_bound(acquisition_config:Config.Acquistion, model_config:Config.NewModel, market_info):
    idx_log_config = log.get_sub_log('indices', model_config, acquisition_config)
    idx_log_config.set_path(acquisition_config)
    new_data_indices = log.load(idx_log_config)
    max_dv = [np.max(market_info['dv'][new_data_indices[c]]) for c in range(model_config.class_number)]    
    return max_dv

def seq_bound(clf, clip_processor, acquisition_config:Config.Acquistion, model_config:Config.NewModel):
    data_log_config = log.get_sub_log('data', model_config, acquisition_config)
    data_log_config.set_path(acquisition_config)
    new_data = log.load(data_log_config)
    new_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                                    num_workers=config['num_workers'])
    new_data_info = CLF.apply_CLF(clf, new_data_loader, clip_processor)
    max_dv = []
    for c in range(model_config.class_number):
        cls_indices = acquistion.extract_class_indices(c, new_data_info['gt'])
        cls_dv = new_data_info['dv'][cls_indices]
        max_dv.append(np.max(cls_dv))
    return max_dv

def get_max_dv(market_info,train_info,n_cls,new_data_indices,pure):
    '''
    Deprecated
    '''
    if pure:
        return [np.max(market_info['dv'][new_data_indices[c]]) for c in range(n_cls)]
    else:
        max_dv = []
        for c in range(n_cls):
            cls_indices = acquistion.extract_class_indices(c, train_info['gt'])
            train_cls_dv = train_info['dv'][cls_indices]
            cls_dv = np.concatenate((train_cls_dv, market_info['dv'][new_data_indices[c]]))
            max_dv.append(np.max(cls_dv))
        return max_dv

def seq_dv_bound(model_config, acquisition_config):
    clf_log = log.get_sub_log('clf', model_config, acquisition_config)
    data_log = log.get_sub_log('data', model_config, acquisition_config)
    clf_data = log.load(clf_log)
    data = log.load(data_log)
    model = Model.load(model_config)
    clf_data_loader ={
        split:torch.utils.data.DataLoader(ds, batch_size=model_config.batch_size, 
                                        num_workers=config['num_workers'])  for split,ds in clf_data.items()
    }
    clf, clip, score = CLF.get_CLF(model, clf_data_loader)
    return score
    # data_loader = torch.utils.data.DataLoader(data, batch_size=model_config.batch_size, 
    #                                     num_workers=config['num_workers'])
    # data_info = apply_CLF(clf, data_loader, clip)
    # max_dv = []
    # for c in range(model_config.class_number):
    #     cls_indices = acquistion.extract_class_indices(c, data_info['gt'])
    #     cls_dv = data_info['dv'][cls_indices]
    #     max_dv.append(np.max(cls_dv))
    # return max_dv
        

