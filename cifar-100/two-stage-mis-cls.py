from utils import *
import matplotlib.pyplot as plt

def main(epochs,new_model_setter='retrain', pure=False):
    print('Use pure: ',pure)
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    sp_labels_map = {
        0: 0,
        3: 1,
        4: 2
    }

    select_fine_labels = [30, 4, 9, 10, 0, 51]
    img_per_cls_list = [75,100,150,200,250] # 3-class

    # select_fine_labels = None
    # if pure:
    #     img_per_cls_list = [200,400,600,850] # all-class-pure
    # else:
    #     img_per_cls_list = [75,100,200,300,400] # all-class

    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    # method_list = ['seq','seq_clf']

    acc_list = []

    for model_cnt in range(epochs):
        print('epoch',model_cnt)
        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        if select_fine_labels is not None:
            for split,infos in ds.items():
                new_ds_split = []
                for info in infos:
                    new_ds_split.append((info[0],sp_labels_map[info[1]],info[2]))
                ds[split] = new_ds_split 
        subset_loader = {
            k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], shuffle=(k=='train'), drop_last=(k=='train'),num_workers=num_workers)
            for k in ds.keys()
        }
        path = os.path.join(model_path_root,'{}.pt'.format(model_cnt))
        base_model = load_model(path)
        base_model = base_model.eval()

        clf,clip_features,_ = get_CLF(base_model,subset_loader)
        gt, pred, conf = evaluate_model(subset_loader['test'],base_model)
        _, dv, _ = clf.predict(gts=gt, latents=clip_features['test'], compute_metrics=False, preds=pred)
        base_acc = (gt==pred).mean()*100
        mis_cls_set_indices = (gt != pred)
        mis_cls_set = torch.utils.data.Subset(ds['test'],np.arange(len(ds['test']))[mis_cls_set_indices])
        mis_cls_loader = torch.utils.data.DataLoader(mis_cls_set, batch_size=hparams['batch_size'], num_workers=num_workers)
        corr_cls_set = torch.utils.data.Subset(ds['test'],np.arange(len(ds['test']))[~mis_cls_set_indices])
        corr_cls_loader = torch.utils.data.DataLoader(corr_cls_set, batch_size=hparams['batch_size'], num_workers=num_workers)
       
        acc_epoch = []
        for method in method_list:
            acc_method = []
            for img_cls in img_per_cls_list:
                pure_name = 'pure' if pure else ''
                new_model_path_root = os.path.join(model_path_root,new_model_setter,pure_name) 
                path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format(method,img_cls,model_cnt))
                new_model = load_model(path)
                new_model = new_model.eval()      

                gt, pred, conf = evaluate_model(corr_cls_loader,base_model)
                base_corr = (gt==pred).sum()
                assert  base_corr== len(gt)
                gt, pred, conf = evaluate_model(mis_cls_loader,new_model)
                acc_img_num = np.concatenate(((gt==pred),np.ones(base_corr))).mean()*100 - base_acc
                # acc_img_num = np.concatenate(((gt==pred),np.ones(base_corr))).mean()*100 # mis_cls_acc = 0
                
                acc_method.append(acc_img_num)
            acc_epoch.append(acc_method)

        acc_list.append(acc_epoch)

    acc_list = np.round(acc_list,decimals=3)
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling','sequential','sequential with only SVM updates']
    # method_labels = ['greedy decision value','random sampling']

    for m_idx,method in enumerate(method_labels): 
        method_result = acc_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
        method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
        plt.plot(img_per_cls_list,method_avg,label=method)
    plt.xticks(img_per_cls_list,img_per_cls_list)
    plt.xlabel('#new images for each superclass')
    plt.ylabel('#model accuracy change')
    plt.legend(fontsize='small')
    pure_name = '-pure' if pure else ''
    fig_name = 'figure/mis-cls/3-class{}.png'.format(pure_name)
    plt.savefig(fig_name)
    plt.clf()
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-p','--pure',type=bool,default=False)

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs, pure=args.pure)
    # print(args.method)