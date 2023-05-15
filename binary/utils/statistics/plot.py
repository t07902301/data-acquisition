import utils.objects.Config as Config
import numpy as np
import os
import matplotlib.pyplot as plt
from abc import abstractmethod

class Prototype():
    def __init__(self, model_config: Config.NewModel) -> None:
        self.model_config = model_config   
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def get_fig_name(self):
        pass

class Line(Prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)

    def get_fig_name(self, data_name):
        fig_root = 'figure/{}'.format(self.model_config.model_dir)
        if os.path.exists(fig_root) is False:
            os.makedirs(fig_root)
        aug_name = '' if self.model_config.augment else 'na'
        return os.path.join(fig_root, '{}-{}.png'.format(aug_name, data_name))
    
    def run(self, value_list, method_list, n_data_list, ylabel, data_name):
        if isinstance(value_list,list):
            value_list = np.array(value_list)        
        for m_idx,method in enumerate(method_list): 
            method_result = value_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
            method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
            plt.plot(n_data_list,method_avg,label=method)
        plt.xticks(n_data_list,n_data_list)
        plt.xlabel('#new images')
        plt.ylabel(ylabel)
        plt.legend(fontsize='small')
        fig_name = self.get_fig_name(data_name)
        plt.savefig(fig_name)
        plt.clf()    
        print('save fig to', fig_name)        

class Histogram(Prototype):
    def __init__(self, model_config: Config.NewModel) -> None:
        super().__init__(model_config)

    def run(self, epochs, dv_list, n_data=None, method=None):
        n_cols = epochs
        split_name = list(dv_list[0].keys())
        n_rows = len(split_name) #n_splits
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, tight_layout=True)
        axs = axs.flatten() #2D -> 1D
        for row in range(n_rows):
            for col in range(n_cols):
                axs[col + n_cols * row].hist(dv_list[col][split_name[row]], bins = 6)
        fig_name = self.get_fig_name(n_data, method)
        fig.suptitle(split_name)
        fig.savefig(fig_name)
        fig.clf()      
        print('save fig to', fig_name)  

    def get_fig_name(self, n_data, method):
        fig_root = 'figure/{}/distribution'.format(self.model_config.model_dir)
        if os.path.exists(fig_root) is False:
            os.makedirs(fig_root)
        if n_data == None and method == None:
            fig_name = os.path.join(fig_root, 'total.png')
        else:
            fig_name = os.path.join(fig_root, '{}-{}.png'.format(method, n_data))
        return fig_name

    # def threshold_collection(self, threshold_list, acc_list):
    #     fig, axs = plt.subplots(2,2, sharey=True, sharex=True)
    #     axs = axs.flatten()
    #     for threshold_idx,threshold in enumerate(threshold_list):
    #         threshold_result = acc_list[:,:,:,threshold_idx]#(e,m,img_num,threshold) e:epochs,m:methods
    #         print('In threshold:',threshold)
    #         for m_idx,method in enumerate(self.plot_methods): 
    #             method_result = threshold_result[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
    #             method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
    #             if threshold_idx == 0 :
    #                 axs[threshold_idx].plot(n_data_list,method_avg,label=method)
    #             else:
    #                 axs[threshold_idx].plot(n_data_list,method_avg)
    #             # axs[threshold_idx].plot(img_per_cls_list,method_avg,label=method)

    #         axs[threshold_idx].set_xticks(n_data_list,n_data_list)
    #         if threshold_idx == 2:
    #             axs[threshold_idx].set_xlabel('#new images for each superclass')
    #             axs[threshold_idx].set_ylabel('#model accuracy change')
    #         axs[threshold_idx].set_title('Threshold:{}'.format(threshold))
    #     fig.legend(loc=2,fontsize='x-small')
    #     fig.tight_layout()
    #     fig.savefig(self.fig_name)
    #     fig.clf()
    