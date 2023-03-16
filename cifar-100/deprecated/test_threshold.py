from utils import *

def run(new_img_num,epoch,method,test_loader,old_model,new_model_setter='refine'):
    threshold_test = test_loader['threshold']
    reamin_test = test_loader['remain']

    gt,pred,_  = evaluate_model(reamin_test,old_model)
    old_correct = (gt==pred)

    new_model_path_root = os.path.join(model_path_root,new_model_setter)
    path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format(method,new_img_num,epoch))
    new_model = load_model(path)
    new_model = new_model.eval()
    gt,pred,_  = evaluate_model(test_loader,new_model)
    gt,pred,_  = evaluate_model(threshold_test,new_model)
    new_correct = (gt==pred)
    total_correct = np.concatenate((old_correct,new_correct))
    return total_correct

def main(epochs):

    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)

    img_per_cls_list = [25,50,75,100]
    # img_per_cls_list = [25,50]
    # method_list = ['dv','sm','conf','mix','seq','seq_clf']
    # method_list = ['seq','seq_clf']
    method_list = ['dv']
    threshold_list = [-0.75,-0.5,-0.25,0]
    # select_fine_labels = [30, 1, 62, 9, 0, 22, 5, 6, 42, 17, 23, 15, 34, 26, 11, 27, 36, 47, 8, 41, 4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    select_fine_labels = None

    acc_list = []
    minority_precision = []
    minority_recall = []
    mis_cls_sb_precision, mis_cls_sb_recall = [], [] 
    base_acc_list = []
    test_set_len = []
    minority_incor = []

    for i in range(epochs):
        print('epoch',i)
        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)

        subset_loader = {
            k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], shuffle=(k=='train'), drop_last=(k=='train'),num_workers=num_workers)
            for k in ds.keys()
        }
        path = os.path.join(model_path_root,'{}.pt'.format(i))
        base_model = load_model(path)
        base_model = base_model.eval()

        clf,clip_features,_ = get_CLF(base_model,subset_loader)
        gt, pred, conf = evaluate_model(subset_loader['test'],base_model)
        _, dv, _ = clf.predict(gts=gt, latents=clip_features['test'], compute_metrics=False, preds=pred)
       
        miscls_mask = gt != pred
        miscls_mask_indices = np.arange(len(miscls_mask))[miscls_mask] # for this class
        miscls_img = torch.utils.data.Subset(ds['test'],miscls_mask_indices)  

        cnt = minority_in_ds(miscls_img)
        mis_cls_sb_precision.append(cnt/len(miscls_img))
        mis_cls_sb_recall.append(cnt/(100*20))

        acc_epoch = []
        minority_precision_epoch = []
        minority_recall_epoch = []
        base_acc_epoch = []
        test_set_len_epoch = []

        minority_incor_epoch = []
        for threshold in threshold_list:
            acc_threshold = []
            dv_selected_indices = dv<threshold
            test_selected = torch.utils.data.Subset(ds['test'],np.arange(len(ds['test']))[dv_selected_indices])
            remained_test = torch.utils.data.Subset(ds['test'],np.arange(len(ds['test']))[~dv_selected_indices])
            test_selected_loader = torch.utils.data.DataLoader(test_selected, batch_size=hparams['batch_size'], num_workers=num_workers)
            remained_test_loader = torch.utils.data.DataLoader(remained_test, batch_size=hparams['batch_size'], num_workers=num_workers)

            test_loader = {
                'threshold':test_selected_loader,
                'remain': remained_test_loader
            }

            cnt = minority_in_ds(test_selected)
            minority_precision_epoch.append(cnt/len(test_selected))
            minority_recall_epoch.append(cnt/(len(remove_fine_labels)*100))

            gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
            base_acc = (gt==pred).mean()*100
            base_acc_epoch.append(base_acc)
            test_set_len_epoch.append(len(test_selected))

            # incor = torch.utils.data.Subset(test_selected,np.arange(len(test_selected))[gt!=pred])
            # cnt_incor = 0
            # for info in incor:
            #     if info[2] in remove_fine_labels:
            #         cnt_incor += 1
            # minority_incor_epoch.append(cnt_incor/cnt)
        

            for method in method_list:
                acc_method = []
                for img_cls in img_per_cls_list:
                    acc = run(img_cls,i,method,test_loader=test_loader,base_model=base_model,new_model_setter='retrain')
                    acc_method.append(acc)
                acc_threshold.append(acc_method)
            acc_epoch.append(acc_threshold)

        acc_list.append(acc_epoch)
        minority_precision.append(minority_precision_epoch)
        minority_recall.append(minority_recall_epoch)
        base_acc_list.append(base_acc_epoch)
        test_set_len.append(test_set_len_epoch)
        # minority_incor.append(minority_incor_epoch)

    # print('real mis_cls_sub')
    # print(np.round(np.mean(minority_incor,axis=0),decimals=3)*100)  
    
    # print('precision avg')
    # print(np.round(np.mean(minority_precision,axis=0),decimals=3)*100)
    # print('recall avg')
    # print(np.round(np.mean(minority_recall,axis=0),decimals=3)*100)  

    # # print('misclassified subclass precision avg')
    # # print(np.round(np.mean(mis_cls_sb_precision),decimals=3)*100)
    # # print('misclassified subclass recall avg')
    # # print(np.round(np.mean(mis_cls_sb_recall),decimals=3)*100)  

    # print('base acc avg')
    # print(np.round(np.mean(base_acc_list,axis=0),decimals=3))

    # print('test set len avg')
    # print(np.round(np.mean(test_set_len,axis=0),decimals=3))

    acc_list = np.round(acc_list,decimals=3)
    print('acc change average')
    for threshold_idx,threshold in enumerate(threshold_list):
        threshold_result = acc_list[:,threshold_idx,:,:]#(e,threshold,m,img_num) e:epochs,m:methods
        print('In threshold:',threshold)
        for m_idx,method in enumerate(method_list): 
            method_result = threshold_result[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
            method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
            print('method:', method)
            print(*method_avg,sep=',')
                

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs)
    # print(args.method)