from src.utils import *

def run(new_img_num,cls_num=2, ds = None,rounds=2,epoch=0,new_model_setter='refine',pure=False, model_path_root='',batch_size=1):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=batch_size, 
                                    shuffle=(k=='train'), drop_last=(k=='train'))
        for k in ds.keys()
    }
    # Load base model
    path = os.path.join(model_path_root,'{}.pt'.format(epoch))
    # path = 'model/base_{}.pt'.format(model_cnt)
    base_model = load_model(path)
    base_model = base_model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    pure_name = 'pure' if pure else ''
    new_model_path_root = os.path.join(model_path_root,new_model_setter,pure_name) 
    if os.path.exists(new_model_path_root) is False:
        os.makedirs(new_model_path_root)

    model = base_model
    market_ds = ds['market']
    train_ds = ds['train']
    new_img_per_round = new_img_num//rounds

    new_img_indices_total = []

    minority_cnt = 0
    for i in range(rounds):

        try:
            # Get SVM 
            clf,clip_features,_ = get_CLF(model,subset_loader) #CLF: a metric of model and data
        except:
            return -100

        # Eval SVM
        market_gt,market_preds,_ = evaluate_model(subset_loader['market'],model)

        # Get decision values
        _, dv, _ = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)
    
        org_market_ds_len = len(market_ds)
        org_train_ds_len = len(train_ds)
        new_img_indices_round = []

        for c in range(cls_num):
            # Get new data
            mask = market_gt==c
            masked_indices = np.arange(len(mask))[mask]
            masked_dv = dv[mask]

            sorted_masked_dv_idx = np.argsort(masked_dv)
            new_img_indices = top_dv(masked_indices[sorted_masked_dv_idx],new_img_per_round,clf=config['clf'])
            new_img_indices_round.append(new_img_indices)

        new_img_indices_round = np.concatenate(new_img_indices_round)
        new_img = torch.utils.data.Subset(ds['market_aug'],new_img_indices_round)
        train_ds = torch.utils.data.ConcatDataset([train_ds,new_img]) 

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
        subset_loader['train'] = train_loader
        assert len(train_ds) == org_train_ds_len + new_img_per_round*cls_num   

        # x_resampled, clf_gt_resampled = SMOTE().fit_resample(x, clf_gt)

        # Exclude new images from the market
        mask_kept_img = np.ones(org_market_ds_len,dtype=bool)
        mask_kept_img[new_img_indices_round] = False
        market_ds = torch.utils.data.Subset(market_ds,np.arange(org_market_ds_len)[mask_kept_img])
        market_loader = torch.utils.data.DataLoader(market_ds, batch_size=batch_size)
        subset_loader['market'] = market_loader      
        assert len(market_ds) == org_market_ds_len - new_img_per_round*cls_num

        # labels = get_ds_labels(train_imgs)
        # weights = get_weights(labels)        
        if new_model_setter == 'retrain':
            model = train_model(subset_loader['train'],subset_loader['val'],num_class=cls_num)
        else:
            model = tune(model,subset_loader['train'],subset_loader['val'])
        
        minority_cnt += minority_in_ds(new_img)
        new_img_indices_total.append(new_img_indices_round) 

    if pure:
        new_img_indices_total = np.concatenate(new_img_indices_total)
        assert len(new_img_indices_total) == new_img_per_round*cls_num*rounds
        new_img_set = torch.utils.data.Subset(ds['market_aug'],new_img_indices_total)
        new_train_loader = torch.utils.data.DataLoader(new_img_set, batch_size=batch_size, shuffle=True,drop_last=True)
        model = train_model(new_train_loader,subset_loader['val'],num_class=cls_num)
    
    model = model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],model)
    retrain_acc = (gt==pred).mean()*100 

    acc_change = np.array(retrain_acc)-np.array(base_acc)

    save_path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format('seq',new_img_num,epoch))
    save_model(model,save_path)

    return acc_change,minority_cnt/(new_img_per_round*cls_num*rounds)

def main(epochs,tune_rounds, pure, model_dir =''):
    print('Use pure: ',pure)

    batch_size = hparams['batch_size'][model_dir]
    select_fine_labels = config['selected_labels'][model_dir]
    label_map = config['label_map'][model_dir]
    model_path_root = os.path.join('model',model_dir,str(batch_size))
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)

    acc_change_list,percent_list = [],[]

    if model_dir.split('-')[0] != '':
        cls_num = int(model_dir.split('-')[0])
    else:
        cls_num = 20

    for i in range(epochs):
        print('in epoch {}'.format(i))
        acc_change_epoch,percent_epoch = [],[]

        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        if select_fine_labels is not None:
            for split,infos in ds.items():
                new_ds_split = []
                for info in infos:
                    new_ds_split.append((info[0],label_map[info[1]],info[2]))
                ds[split] = new_ds_split 

        for img_per_cls in config['acquired_num_per_class']:
            acc_change,percent = run(img_per_cls,cls_num=cls_num,ds=ds,rounds=tune_rounds,epoch=i,new_model_setter='retrain', pure=pure, model_path_root=model_path_root,batch_size=batch_size)
            acc_change_epoch.append(acc_change)
            percent_epoch.append(percent)

        acc_change_list.append(acc_change_epoch)   
        percent_list.append(percent_epoch)

    print('acc change')
    acc_change_list = np.round(acc_change_list,decimals=3)
    print(*acc_change_list.tolist())
    print('acc change average')
    print(*np.round(np.mean(acc_change_list,axis=0),decimals=3),sep=',')

    print('percent')
    print(*np.round(np.mean(percent_list,axis=0)*100,decimals=3),sep=',')

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-r','--rounds',type=int,default=2)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    # parser.add_argument('-b','--batches',type=int,default=1)
    # parser.add_argument('-s','--segments',type=int,default=1)
    # parser.add_argument('-bt','--bottom',type=bool,default=True)
    # parser.add_argument('-s','--stride',type=int,default=0)


    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.rounds, args.pure,args.model_dir)
    # print(args.method)