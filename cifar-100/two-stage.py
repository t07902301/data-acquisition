from utils.strategy import *
import matplotlib.pyplot as plt
import os
import pickle
class test_subet_setter():
    def __init__(self, method:str, aux:list) -> None:
        self.method = method
        self.auxiliary = aux
    def get_subset(self):
        if self.method == 'threshold':
            return self.threshold()
        else:
            return self.misclassification()
    def get_dataset_info(self, info:dict):
        self.ds_info = info
    def threshold(self):
        subset_loader = []
        for threshold in self.auxiliary:
            dv_selected_indices = self.ds_info['dv']<threshold
            test_selected = torch.utils.data.Subset( self.ds_info['dataset'],np.arange(len(self.ds_info['dataset']))[dv_selected_indices])
            remained_test = torch.utils.data.Subset(self.ds_info['dataset'],np.arange(len(self.ds_info['dataset']))[~dv_selected_indices])
            test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=self.ds_info['batch_size'], num_workers=config['num_workers'])
            remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=self.ds_info['batch_size'], num_workers=config['num_workers'])
            test_loader = {
                'new_model':test_selected_loader,
                'old_model': remained_test_loader
            }   
            subset_loader.append(test_loader)
            # assert len(test_selected)+len(remained_test) == len(self.ds_info['dataset'])
        return subset_loader
    def misclassification(self):
        incorr_cls_indices = (self.ds_info['gt'] != self.ds_info['pred'])
        corr_cls_indices = (self.ds_info['gt'] == self.ds_info['pred'])
        incorr_cls_set = torch.utils.data.Subset( self.ds_info['dataset'],np.arange(len(self.ds_info['dataset']))[incorr_cls_indices])
        corr_cls_set = torch.utils.data.Subset( self.ds_info['dataset'],np.arange(len(self.ds_info['dataset']))[corr_cls_indices])
        corr_cls_loader = torch.utils.data.DataLoader(corr_cls_set, batch_size=self.ds_info['batch_size'], num_workers=config['num_workers'])
        incorr_cls_loader = torch.utils.data.DataLoader(incorr_cls_set, batch_size=self.ds_info['batch_size'], num_workers=config['num_workers'])
        test_loader = {
            'new_model':incorr_cls_loader,
            'old_model': corr_cls_loader
        }   
        subset_loader = [test_loader]
        return subset_loader
    # def plot_statistics(self,acc):
    #     if self.method=='threshold':


def run(ds, old_model_config:OldModelConfig, new_model_config:NewModelConfig, acquisition_config:AcquistionConfig, method_list, new_img_num_list, ds_setter:test_subet_setter):
    data_loader = get_dataloader(ds, new_model_config.batch_size)
    # Load base model
    base_model = load_model(old_model_config.path)
    # Get SVM 
    clf,clip_features,_ = get_CLF(base_model,data_loader)

    test_info = apply_CLF(clf,base_model,data_loader,clip_features, 'test')
    base_acc = (test_info['gt']==test_info['pred']).mean()*100

    test_info['batch_size'] = old_model_config.batch_size
    test_info['dataset'] = ds['test']

    ds_setter.get_dataset_info(test_info)
    test_loaders = ds_setter.get_subset()
    acc_change = []
    for method in method_list:
        acc_method = []
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            new_model_config.set_path(acquisition_config)
            new_model = load_model(new_model_config.path)
            acc_img = []
            for loader in test_loaders:

                gt,pred,_  = evaluate_model(loader['new_model'],new_model)
                new_correct = (gt==pred)

                gt,pred,_  = evaluate_model(loader['old_model'],base_model)
                old_correct = (gt==pred)

                total_correct = np.concatenate((old_correct,new_correct))
                assert total_correct.size == test_info['gt'].size
                acc_img.append(total_correct.mean()*100-base_acc)
                # acc_img.append(new_correct.mean()*100-old_correct.mean()*100)

            acc_method.append(acc_img)
        acc_change.append(acc_method)
    return acc_change

def main(epochs,new_model_setter='retrain', pure=False, model_dir ='', subset_method='mis'):
    print('Use pure: ',pure)
    print(subset_method)
    hparams = config['hparams']
    batch_size = hparams['batch_size'][model_dir]

    data_config = config['data']
    select_fine_labels = data_config['selected_labels'][model_dir]
    label_map = data_config['label_map'][model_dir]

    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    # method_list = ['dv']
    threshold_list = [-0.75,-0.5,-0.25,0]
    # threshold_list = [-0.75]

    acc_list = []
    # img_per_cls_list = [75]
    img_per_cls_list = data_config['acquired_num_per_class']['mini'] if 'mini' in model_dir else data_config['acquired_num_per_class']['non-mini']

    superclass_num = int(model_dir.split('-')[0])

    for model_cnt in range(epochs):
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, model_cnt)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, pure, new_model_setter)
        acquistion_config = AcquistionConfig(model_cnt=model_cnt, sequential_rounds= 0)

        print('epoch',model_cnt)
        ds = create_dataset_split(data_config['ds_root'],select_fine_labels,model_dir)

        if select_fine_labels!= []:
            modify_coarse_label(ds,label_map)

        test_subset = test_subet_setter(subset_method,threshold_list)

        acc_epoch = run(ds, old_model_config, new_model_config, acquistion_config, method_list, img_per_cls_list, test_subset)

        acc_list.append(acc_epoch)

    acc_list = np.round(acc_list,decimals=3)
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling', 'sequential', 'sequential with only SVM updates']
    # method_labels = ['greedy decision value']

    fig, axs = plt.subplots(2,2, sharey=True, sharex=True)
    axs = axs.flatten()
    for threshold_idx,threshold in enumerate(threshold_list):
        threshold_result = acc_list[:,:,:,threshold_idx]#(e,m,img_num,threshold) e:epochs,m:methods
        print('In threshold:',threshold)
        for m_idx,method in enumerate(method_labels): 
            method_result = threshold_result[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
            method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
            if threshold_idx == 0 :
                axs[threshold_idx].plot(img_per_cls_list,method_avg,label=method)
            else:
                axs[threshold_idx].plot(img_per_cls_list,method_avg)
            # axs[threshold_idx].plot(img_per_cls_list,method_avg,label=method)

        axs[threshold_idx].set_xticks(img_per_cls_list,img_per_cls_list)
        if threshold_idx == 2:
            axs[threshold_idx].set_xlabel('#new images for each superclass')
            axs[threshold_idx].set_ylabel('#model accuracy change')
        axs[threshold_idx].set_title('Threshold:{}'.format(threshold))
    fig.legend(loc=2,fontsize='x-small')
    fig.tight_layout()
    pure_name = '-pure' if pure else ''
    fig_name = 'figure/{}{}{}.png'.format(model_dir,pure_name,subset_method)
    fig.savefig(fig_name)
    fig.clf()
    print('save to', fig_name)
    
    for m_idx,method in enumerate(method_labels): 
        method_result = acc_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
        method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
        plt.plot(img_per_cls_list,method_avg,label=method)
    plt.xticks(img_per_cls_list,img_per_cls_list)
    plt.xlabel('#new images for each superclass')
    plt.ylabel('#model accuracy change')
    plt.legend(fontsize='small')
    pure_name = '-pure' if pure else ''
    # fig_name = 'figure/mis-cls/3-class{}.png'.format(pure_name)
    pure_name = '-pure' if pure else ''
    fig_name = 'figure/mis-{}{}.png'.format(model_dir,pure_name)
    plt.savefig(fig_name)
    plt.clf()    

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir)
    # print(args.method)