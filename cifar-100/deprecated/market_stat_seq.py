from utils import *
import scipy

def run(new_img_num,cls_num=2, ds = None,rounds=2,epoch=0,new_model_setter='refine'):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], 
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

    new_model_path_root = os.path.join(model_path_root,new_model_setter)
    if os.path.exists(new_model_path_root) is False:
        os.makedirs(new_model_path_root)

    model = base_model
    market_ds = ds['market']
    train_ds = ds['train']

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
        new_img_indices_total = []

        for c in range(cls_num):
            # Get new data
            mask = market_gt==c
            masked_indices = np.arange(len(mask))[mask]
            masked_dv = dv[mask]

            sorted_masked_dv_idx = np.argsort(masked_dv)
            new_img_indices = top_dv(masked_indices[sorted_masked_dv_idx],new_img_num//rounds,clf=config['clf'])
            new_img_indices_total.append(new_img_indices)

        new_img_indices_total = np.concatenate(new_img_indices_total)
        new_img = torch.utils.data.Subset(ds['market_aug'],new_img_indices_total)
        train_ds = torch.utils.data.ConcatDataset([train_ds,new_img])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True,drop_last=True)
        subset_loader['train'] = train_loader
        assert len(train_ds) == org_train_ds_len + new_img_num//rounds*cls_num   

        # x_resampled, clf_gt_resampled = SMOTE().fit_resample(x, clf_gt)

        # Exclude new images from the market
        mask_kept_img = np.ones(org_market_ds_len,dtype=bool)
        mask_kept_img[new_img_indices_total] = False
        market_ds = torch.utils.data.Subset(market_ds,np.arange(org_market_ds_len)[mask_kept_img])
        market_loader = torch.utils.data.DataLoader(market_ds, batch_size=hparams['batch_size'])
        subset_loader['market'] = market_loader      
        assert len(market_ds) == org_market_ds_len - new_img_num//rounds*cls_num

        # labels = get_ds_labels(train_imgs)
        # weights = get_weights(labels)        
        if new_model_setter == 'retrain':
            model = train_model(subset_loader['train'],subset_loader['val'],num_class=cls_num)
        else:
            model = tune(model,subset_loader['train'],subset_loader['val'])
        
        minority_cnt += minority_in_ds(new_img)

    return minority_cnt/(new_img_num//rounds*cls_num*rounds)

def main(epochs,img_per_cls,tune_rounds):
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)

    img_per_cls_list = [25,50,75,100]

    percent = []
    sp_labels_map = {
        0: 0,
        3: 1,
        4: 2
    }
    # select_fine_labels = [30, 1, 62, 9, 0, 22, 5, 6, 42, 17, 23, 15, 34, 26, 11, 27, 36, 47, 8, 41, 4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    # select_fine_labels = None
    select_fine_labels = [30, 4, 9, 10, 0, 51]

    cls_num = 20 if select_fine_labels==None else len(select_fine_labels)//2

    for i in range(epochs):
        print('in epoch {}'.format(i))
        percent_epoch = []

        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        if select_fine_labels is not None:
            for split,infos in ds.items():
                new_ds_split = []
                for info in infos:
                    new_ds_split.append((info[0],sp_labels_map[info[1]],info[2]))
                ds[split] = new_ds_split 

        for img_per_cls in img_per_cls_list:
            minority_percent = run(img_per_cls,cls_num=cls_num,ds=ds,rounds=tune_rounds,epoch=i,new_model_setter='retrain')
            percent_epoch.append(minority_percent)
        percent.append(percent_epoch)   

    print('percent')
    print(percent)
    print(*np.round(np.mean(percent,axis=0)*100,decimals=3),sep=',')
    print(hparams)
    print(select_fine_labels)
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    parser.add_argument('-c','--img_per_cls',help='images to be added per class',type=int)
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-r','--rounds',type=int,default=2)

    # parser.add_argument('-b','--batches',type=int,default=1)
    # parser.add_argument('-s','--segments',type=int,default=1)
    # parser.add_argument('-bt','--bottom',type=bool,default=True)
    # parser.add_argument('-s','--stride',type=int,default=0)


    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.img_per_cls,args.rounds)
    # print(args.method)