import matplotlib.pyplot as plt
from utils import config
from utils.Config import *
from utils.log import *
from utils.acquistion import apply_CLF,get_CLF, get_class_info
from utils.model import *
from abc import abstractmethod
import numpy as np
class test_subet_setter():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_subset_loders(self):
        pass
class threshold_test_subet_setter(test_subet_setter):
    def __init__(self) -> None:
        pass
    def get_subset_loders(self, data_info, threshold):
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices, cls_mask, cls_dv = get_class_info(c, data_info['gt'], data_info['dv'])
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
def subset_setter_factory(method,thresholds):
    if method == 'mis':
        return misclassification_test_subet_setter()
    else:
        return threshold_test_subet_setter(thresholds)
class plotter():
    def __init__(self, select_method, plot_methods, plot_data_numbers, model_config) -> None:
        self.select_method = select_method
        self.plot_methods = plot_methods
        self.plot_data_numbers = plot_data_numbers
        pure_name = '-pure' if model_config.pure else ''
        self.fig_name = 'figure/{}{}-{}.png'.format(model_config.model_dir,pure_name,self.select_method)

    def threshold_plot(self, threshold_list, acc_list ):
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

    def other_plot(self, value_list):
        if self.select_method == 'dv':
            ylabel = 'decision value'
        elif self.select_method == 'total':
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

    def simple_threshold_plot(self,acc_list):
        for m_idx,method in enumerate(self.plot_methods): 
            method_result = acc_list[:,:,m_idx] #(e,img_num,m) e:epochs,m:methods
            method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
            plt.plot(self.plot_data_numbers,method_avg,label=method)
        plt.xticks(self.plot_data_numbers,self.plot_data_numbers)
        plt.xlabel('#new images for each superclass')
        plt.ylabel('model accuracy change')
        plt.legend(fontsize='small')
        plt.savefig(self.fig_name)
        plt.clf()   
    def plot_data(self,acc_list,threshold_list):
        # if self.select_method == 'threshold':
        #     self.threshold_plot(threshold_list, acc_list)
        # elif self.select_method != 'bm': 
        #     self.other_plot(acc_list)
        # else:
        #     print(acc_list)
        if isinstance(acc_list,list):
            acc_list = np.array(acc_list)
        if self.select_method == 'threshold':
            self.simple_threshold_plot(acc_list)
        else:
            self.other_plot(acc_list)
        print('save to', self.fig_name)        

class checker():
    def __init__(self,model_config:NewModelConfig) -> None:
        self.model_config = model_config
    @abstractmethod
    def run(self,acquisition_config:AcquistionConfig):
        pass
    @abstractmethod
    def setup(self, old_model_config:OldModelConfig, data_splits):
        pass
class dv_checker(checker):
    def __init__(self, model_config: NewModelConfig) -> None:
        super().__init__(model_config)
        # checker.__init__(self, model_config) # TODO: a problem in the constructor of multi-inheritence MRO
        self.log_config = LogConfig(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter)
    def load_log(self):
        data = torch.load(self.log_config.path)
        print('load DV from {}'.format(self.log_config.path))
        return data
    def setup(self, old_model_config, datasplits):
        self.base_model = load_model(old_model_config.path)
        clf,clip_processor,_ = get_CLF(self.base_model,datasplits.loader)
        self.clf = clf
        self.clip_processor = clip_processor   
    def run(self,acquisition_config):
        self.log_config.set_path(acquisition_config)
        data = self.load_log()
        loader = torch.utils.data.DataLoader(data, batch_size=self.model_config.batch_size, shuffle=True,drop_last=True)
        data_info = apply_CLF(self.clf, loader, self.clip_processor, self.base_model)
        return np.round(np.mean(data_info['dv']),decimals=3)
class benchmark_checker(dv_checker):
    def __init__(self, model_config: NewModelConfig) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        super().setup(old_model_config, datasplits)
        self.datasplits = datasplits       
    def run(self):
        data_info = apply_CLF(self.clf, self.datasplits.loader['market'], self.clip_processor, self.base_model)
        # return (data_info['dv']<0).sum()/len(data_info['dv'])     
        return np.std(data_info['dv']), np.mean(data_info['dv'])
class test_subset_checker(checker):
    def __init__(self, model_config: NewModelConfig, threshold_collection) -> None:
        super().__init__(model_config)
        self.threshold_collection = threshold_collection
    def setup(self, old_model_config, datasplits):
        self.base_model = load_model(old_model_config.path)
        # state = np.random.get_state()
        clf,clip_processor,_ = get_CLF(self.base_model,datasplits.loader)
        test_info = apply_CLF(clf,datasplits.loader['test'],clip_processor)
        test_info['batch_size'] = old_model_config.batch_size
        test_info['dataset'] = datasplits.dataset['test']
        test_info['loader'] = datasplits.loader['test']
        self.test_info = test_info
        gt, pred, _ = evaluate_model( datasplits.loader['test'], self.base_model)
        self.base_acc = (gt == pred).mean()*100   
        # np.random.set_state(state) 
    def set_test_loader(self,loader):
        self.test_info['loader'] = loader
    def run(self, acquistion_config:AcquistionConfig):
        self.model_config.set_path(acquistion_config=acquistion_config)
        return self.target_test(self.test_info['loader'])

    def iter_test(self):
        new_model = load_model(self.model_config.path)
        acc_change = []
        for threshold in self.threshold_collection:
            loader = threshold_test_subet_setter().get_subset_loders(self.test_info,threshold)
            gt,pred,_  = evaluate_model(loader['new_model'],new_model)
            new_correct = (gt==pred)
            gt,pred,_  = evaluate_model(loader['old_model'], self.base_model)
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
            assert total_correct.size == self.test_info['gt'].size
            acc_change.append(total_correct.mean()*100-self.base_acc)        
        return acc_change
    def target_test(self, loader):
        new_model = load_model(self.model_config.path)
        gt,pred,_  = evaluate_model(loader['new_model'],new_model)
        new_correct = (gt==pred)
        gt,pred,_  = evaluate_model(loader['old_model'], self.base_model)
        old_correct = (gt==pred)
        total_correct = np.concatenate((old_correct,new_correct))
        assert total_correct.size == self.test_info['gt'].size   
        return total_correct.mean()*100-self.base_acc 

class test_set_checker(checker):
    def __init__(self, model_config: NewModelConfig) -> None:
        super().__init__(model_config)
    def setup(self, old_model_config, datasplits):
        self.base_model = load_model(old_model_config.path)
        self.test_loader = datasplits.loader['test']
        gt, pred, _ = evaluate_model(self.test_loader,self.base_model)
        self.base_acc = (gt==pred).mean()*100

    def run(self, acquisition_config):
        self.model_config.set_path(acquisition_config)
        new_model = load_model(self.model_config.path)
        gt, pred, _ = evaluate_model(self.test_loader,new_model)
        acc = (gt==pred).mean()*100
        return acc - self.base_acc

def checker_factory(check_method,model_config, threshold_collection):
    if check_method == 'dv':
        checker_product = dv_checker(model_config)
    elif check_method == 'total':
        checker_product = test_set_checker(model_config)
    elif check_method == 'bm':
        checker_product = benchmark_checker(model_config)
    else:
        checker_product = test_subset_checker(model_config, threshold_collection) 
    return checker_product

def get_threshold_collection(data_number_list, acquisition_config:AcquistionConfig, model_config:NewModelConfig, old_model_config:OldModelConfig, data_splits, model_cnt):
    idx_log_config = LogConfig(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter,model_cnt=model_cnt)
    idx_log_config.root = os.path.join(idx_log_config.root, 'indices')
    base_model = load_model(old_model_config.path)

    threshold_dict = {}
    for data_number in data_number_list:
        data_splits.get_dataloader(model_config.batch_size)
        clf,clip_processor,_ = get_CLF(base_model,data_splits.loader)
        market_info = apply_CLF(clf,data_splits.loader['market'],clip_processor)
        acquisition_config.set_items('dv', data_number)
        idx_log_config.set_path(acquisition_config)
        indices = load_log(idx_log_config.path)
        max_dv = [np.max(market_info['dv'][indices[c]]) for c in range(model_config.class_number)]
        threshold_dict[data_number] = max_dv
    # print(threshold_dict)
    return threshold_dict




