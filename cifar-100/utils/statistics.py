import matplotlib.pyplot as plt
from utils import *
from utils.Config import *
from utils.acquistion import apply_CLF,get_CLF
from utils.model import *
from abc import abstractmethod
class test_subet_setter():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def get_subset_loders(self):
        pass
class threshold_test_subet_setter(test_subet_setter):
    def __init__(self, thresholds: list) -> None:
        self.thresholds = thresholds
    def get_subset_loders(self, data_info):
        subset_loader = []
        for threshold in self.thresholds:
            dv_selected_indices = data_info['dv']<threshold
            test_selected = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[dv_selected_indices])
            remained_test = torch.utils.data.Subset(data_info['dataset'],np.arange(len(data_info['dataset']))[~dv_selected_indices])
            test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
            remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=data_info['batch_size'], num_workers=config['num_workers'])
            test_loader = {
                'new_model':test_selected_loader,
                'old_model': remained_test_loader
            }   
            subset_loader.append(test_loader)
            print('in {}, subset size: {}'.format(threshold, len(test_selected)))
            # assert len(test_selected)+len(remained_test) == len(data_info['dataset'])
        return subset_loader
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

    def threshold_plot(self, threshold_list, acc_list, ):
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

    def plot_data(self,acc_list,threshold_list):
        if self.select_method == 'threshold':
            self.threshold_plot(threshold_list, acc_list)
        elif self.select_method != 'bm': 
            self.other_plot(acc_list)
        else:
            print(acc_list)
        print('save to', self.fig_name)        



class checker():
    def __init__(self,model_config:NewModelConfig) -> None:
        self.model_config = model_config
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def setup(self):
        pass
class dv_checker(checker):
    def __init__(self, model_config: NewModelConfig) -> None:
        super().__init__(model_config)
        self.log_config = LogConfig(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter)
    def load_log(self):
        data = torch.load(self.log_config.path)
        print('load from {}'.format(self.log_config.path))
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
    def run(self,acquisition_config):
        data_info = apply_CLF(self.clf, self.datasplits.loader['market'], self.clip_processor, self.base_model)
        # return (data_info['dv']<0).sum()/len(data_info['dv'])     
        return np.std(data_info['dv']), np.mean(data_info['dv'])
class test_subset_checker(checker):
    def __init__(self, model_config: NewModelConfig, threshold_list:list, method) -> None:
        super().__init__(model_config)
        self.test_subset = subset_setter_factory(method,threshold_list)
    def setup(self, old_model_config, datasplits):
        self.base_model = load_model(old_model_config.path)
        clf,clip_processor,_ = get_CLF(self.base_model,datasplits.loader)
        test_info = apply_CLF(clf,datasplits.loader['test'],clip_processor,self.base_model)
        test_info['batch_size'] = old_model_config.batch_size
        test_info['dataset'] = datasplits.dataset['test']
        test_info['loader'] = datasplits.loader['test']
        self.test_info = test_info
        self.base_acc = (test_info['gt']==test_info['pred']).mean()*100    
        self.test_loaders = self.test_subset.get_subset_loders(self.test_info)

    def run(self, acquistion_config):
        self.model_config.set_path(acquistion_config=acquistion_config)
        new_model = load_model(self.model_config.path)
        acc_change = []
        for loader in self.test_loaders:
            gt,pred,_  = evaluate_model(loader['new_model'],new_model)
            new_correct = (gt==pred)
            gt,pred,_  = evaluate_model(loader['old_model'], self.base_model)
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
            assert total_correct.size == self.test_info['gt'].size
            acc_change.append(total_correct.mean()*100-self.base_acc)        
        return acc_change
    
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

def checker_factory(check_method,model_config, threshold_list):
    if check_method == 'dv':
        checker_product = dv_checker(model_config)
    elif check_method == 'total':
        checker_product = test_set_checker(model_config)
    elif check_method == 'bm':
        checker_product = benchmark_checker(model_config)
    else:
        checker_product = test_subset_checker(model_config, threshold_list, check_method) 
    return checker_product