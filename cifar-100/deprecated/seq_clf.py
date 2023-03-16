from src.utils import *

def run(new_img_num,cls_num=2, ds = None,model_cnt=0,rounds=2,new_model_setter='retrain', pure=False, model_path_root='',batch_size=1):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=batch_size, 
                                    shuffle=(k=='train'), drop_last=(k=='train'))
        for k in ds.keys()
    }
    # Load base model
    path = os.path.join(model_path_root,'{}.pt'.format(model_cnt))
    # path = 'model/base_{}.pt'.format(model_cnt)
    base_model = load_model(path)
    base_model = base_model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    val_ds = ds['val']
    market_ds = ds['market']
    train_ds = ds['train']
    org_val_ds = ds['val']

    try:
        clf,clip_features,_ = get_CLF(base_model,subset_loader)
    except:
        return -100

    minority_cnt = 0
    new_img_indices_total = []

    for i in range(rounds):

        # Eval SVM
        market_gt,market_preds,_ = evaluate_model(subset_loader['market'],base_model)

        # Get new data
        _, dv, _ = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)
        org_market_ds_len = len(market_ds)
        org_train_ds_len = len(train_ds)
        org_val_ds_len = len(val_ds)
        new_img_per_round = new_img_num // rounds
        new_img_indices_round = []
        for c in range(cls_num):
            cls_mask = market_gt==c
            market_indices = np.arange(len(market_gt))
            cls_market_indices = market_indices[cls_mask]
            cls_dv = dv[cls_mask] # cls_dv and cls_market_indices are mutually dependent, cls_market_indices index to locations of each image and its decision value
            sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
            new_img_indices = top_dv(cls_market_indices[sorted_idx],new_img_per_round,clf=config['clf'])
            new_img_indices_round.append(new_img_indices)
        
        new_img_indices_round = np.concatenate(new_img_indices_round)
        new_img = torch.utils.data.Subset(ds['market_aug'],new_img_indices_round)


        mask_kept_img = np.ones(org_market_ds_len,dtype=bool)
        mask_kept_img[new_img_indices_round] = False
        market_ds = torch.utils.data.Subset(market_ds,np.arange(org_market_ds_len)[mask_kept_img])

        market_loader = torch.utils.data.DataLoader(market_ds, batch_size=batch_size)
        subset_loader['market'] = market_loader      
        assert len(market_ds) == org_market_ds_len - new_img_per_round*cls_num

        val_ds = torch.utils.data.ConcatDataset([val_ds,new_img])
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        subset_loader['val'] = val_loader      
        assert len(val_ds) == org_val_ds_len + new_img_per_round*cls_num   

        try:
            # update SVM
            clf,clip_features,_ = get_CLF(base_model,subset_loader)
        except:
            return -100
        minority_cnt += minority_in_ds(new_img)
        new_img_indices_total.append(new_img_indices_round)

    new_img_indices_total = np.concatenate(new_img_indices_total)
    assert len(new_img_indices_total) == new_img_per_round*cls_num*rounds
    new_img_set = torch.utils.data.Subset(ds['market_aug'],new_img_indices_total)
    if pure:
        new_train_set = new_img_set
    else:
        new_train_set = torch.utils.data.ConcatDataset([train_ds,new_img_set])
        assert len(new_train_set) == org_train_ds_len + new_img_per_round*cls_num*rounds  
    new_train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    org_val_loader = torch.utils.data.DataLoader(org_val_ds, batch_size=batch_size)
    assert len(org_val_ds) == len(val_ds) - new_img_per_round*cls_num*rounds 

    pure_name = 'pure' if pure else ''
    new_model_path_root = os.path.join(model_path_root,new_model_setter,pure_name)
    if os.path.exists(new_model_path_root) is False:
        os.makedirs(new_model_path_root)

    model = train_model(new_train_loader,org_val_loader)
    save_path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format('seq_clf',new_img_num,model_cnt))
    save_model(model,save_path)

    # model = get_tuned_model(base_model,new_train_loader,subset_loader['val'])
    model = model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],model)
    retrain_acc = (gt==pred).mean()*100 
    acc_change = np.array(retrain_acc)-np.array(base_acc)
    return acc_change, minority_cnt/(new_img_per_round*cls_num*rounds)

def main(epochs,tune_rounds,pure, model_dir =''):
    batch_size = hparams['batch_size'][model_dir]
    select_fine_labels = config['selected_labels'][model_dir]
    label_map = config['label_map'][model_dir]
    model_path_root = os.path.join('model',model_dir,str(batch_size))
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    if model_dir.split('-')[0] != '':
        cls_num = int(model_dir.split('-')[0])
    else:
        cls_num = 20
    acc_change_list, percent_list = [], []
    
    for i in range(epochs):
        print('in epoch {}'.format(i))
        acc_change_epoch, percent_epoch = [], []

        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        if select_fine_labels is not None:
            for split,infos in ds.items():
                new_ds_split = []
                for info in infos:
                    new_ds_split.append((info[0],label_map[info[1]],info[2]))
                ds[split] = new_ds_split 

        for img_per_cls in config['acquired_num_per_class']:
            acc_change, percent =  run(img_per_cls,cls_num=cls_num,ds=ds,model_cnt=i,rounds=tune_rounds,new_model_setter='retrain',pure=pure,model_path_root=model_path_root,batch_size=batch_size)
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
    print(*np.round(np.mean(percent_list,axis=0),decimals=3)*100,sep=',')


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
    main(args.epochs, args.rounds, args.pure,args.model_dir)

    # print(args.method)