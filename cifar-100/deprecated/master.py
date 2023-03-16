from src.utils import *
# import scipy

def run(cls_num,ds,methods,new_img_num_list,epoch=0,new_model_setter='refine',pure=False,model_path_root='',batch_size=1):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=batch_size, 
                                    shuffle=(k=='train'), drop_last=(k=='train'),num_workers=num_workers)
        for k in ds.keys()
    }
    # Load base model
    path = os.path.join(model_path_root,'{}.pt'.format(epoch))
    base_model = load_model(path)
    base_model = base_model.eval()    

    gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    try:
        # Get SVM 
        clf,clip_features,_ = get_CLF(base_model,subset_loader)

    except Exception as  e:
        print(e)
        return [[-100 for i in range(new_img_num_list)] for j in range(len(methods))]

    # Eval SVM
    market_gt,market_preds, market_conf = evaluate_model(subset_loader['market'],base_model)

    # Get decision values
    _, dv, _ = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)

    acc_change_epoch = []

    percent_epoch = []

    pure_name = 'pure' if pure else ''
    new_model_path_root = os.path.join(model_path_root,new_model_setter,pure_name) 
    if os.path.exists(new_model_path_root) is False:
        os.makedirs(new_model_path_root)
        
    for method in methods:

        print('In method', method)
        acc_change_method = []
        for new_img_num in new_img_num_list:
            model = deepcopy(base_model)
            train_ds = ds['train']
            org_train_ds_len = len(train_ds)
            new_img_set_indices = []

            for c in range(cls_num):
                cls_mask = market_gt==c
                market_indices = np.arange(len(market_gt))
                cls_market_indices = market_indices[cls_mask]
                cls_dv = dv[cls_mask] # cls_dv and cls_market_indices are mutually dependent, cls_market_indices index to locations of each image and its decision value

                if (method == 'hard') or (method=='easy'):
                    cls_new_img_indices = dummy_acquire(cls_gt=market_gt[cls_mask],cls_pred=market_preds[cls_mask],method=method, img_num=new_img_num)
                    new_img_indices = cls_market_indices[cls_new_img_indices]
                 
                    # visualize_images(class_imgs,new_img_indices,cls_dv[new_img_indices],path=os.path.join('vis',method))
                else:
                    if method == 'dv':
                        sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
                        new_img_indices = top_dv(cls_market_indices[sorted_idx],new_img_num,clf=config['clf'])
                    elif method == 'sm':
                        new_img_indices = np.random.choice(cls_market_indices,new_img_num,replace=False) # points to the origin dataset
                    elif method == 'mix':
                        sorted_idx = np.argsort(cls_dv)
                        new_img_indices = top_dv(cls_market_indices[sorted_idx],new_img_num-new_img_num//2)
                        new_img_indices = np.concatenate((new_img_indices,np.random.choice(cls_market_indices,new_img_num//2,replace=False))) 
                    elif method == 'conf':
                        masked_conf = market_conf[cls_mask]
                        sorted_idx = np.argsort(masked_conf) # uncertain points
                        new_img_indices = top_dv(cls_market_indices[sorted_idx],new_img_num,clf=config['clf'])

                new_img_set_indices.append(new_img_indices)

            new_img_set_indices = np.concatenate(new_img_set_indices)
            new_img_set = torch.utils.data.Subset(ds['market_aug'],new_img_set_indices)
            train_ds = new_img_set if pure else torch.utils.data.ConcatDataset([train_ds,new_img_set])

            if method not in ['hard','easy']:
                if pure:
                    assert len(train_ds) == new_img_num*cls_num
                else:
                    # print(len(train_ds),org_train_ds_len,new_img_num,cls_num)
                    assert len(train_ds) == org_train_ds_len + new_img_num*cls_num

            new_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)

            if new_model_setter=='refine':
                new_model = tune(model,new_train_loader,subset_loader['val']) # tune
            else:
                new_model = train_model(new_train_loader,subset_loader['val'],num_class=cls_num) # retrain

            save_path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format(method,new_img_num,epoch))
            save_model(new_model,save_path)

            # Eval tuned model
            new_model = new_model.eval()
            gt,pred,_  = evaluate_model(subset_loader['test'],new_model)
            new_acc = (gt==pred).mean()*100 

            # visualize_images(market_ds,new_img_indices,dv,path='log/market/tune/init/figures/{}.png'.format(model_cnt))
            acc_change = new_acc - base_acc
            acc_change_method.append(acc_change)
            del new_model
            del model

        acc_change_epoch.append(acc_change_method)

    return acc_change_epoch, percent_epoch

def main(epochs, new_model_setter='retrain', pure=False,model_dir =''):
    print('Use pure: ',pure)

    batch_size = hparams['batch_size'][model_dir]
    select_fine_labels = config['selected_labels'][model_dir]
    label_map = config['label_map'][model_dir]
    model_path_root = os.path.join('model',model_dir,str(batch_size))

    acc_change_list = []
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)

    # select_fine_labels = None
    # # new_img_num_list = [150,250,300] # all-class
    # # new_img_num_list = [300] # all-class
    # new_img_num_list = [75,100,200,300] # all-class

    # new_img_num_list = [75,100,150,200,250] # 3-class
    # select_fine_labels = [30, 4, 9, 10, 0, 51]
  
    method_list = ['dv','sm','conf','mix']
    # method_list = ['dv']
    # method_list = ['hard', 'easy','dv','sm','conf','mix']
    # method_list = ['hard', 'easy']
   
    acc_change_list = []
    percent_list = []
    if model_dir.split('-')[0] != '':
        cls_num = int(model_dir.split('-')[0])
    else:
        cls_num = 20
    # cls_num = 20 if select_fine_labels==None else len(select_fine_labels)//2

    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        
        if select_fine_labels is not None:
            for split,infos in ds.items():
                new_ds_split = []
                for info in infos:
                    new_ds_split.append((info[0],label_map[info[1]],info[2]))
                ds[split] = new_ds_split 

        acc_change_epoch, percent_epoch = run(cls_num=cls_num,methods=method_list,new_img_num_list=config['acquired_num_per_class'],ds=ds,epoch=epo,new_model_setter='retrain',pure=pure,model_path_root=model_path_root,batch_size=batch_size)
        acc_change_list.append(acc_change_epoch)
        percent_list.append(percent_epoch)

    print('acc change')
    acc_change_list = np.round(acc_change_list,decimals=3)
    print(*acc_change_list.tolist(),sep='\n')
    print('acc change average')
    for m_idx,method in enumerate(method_list): 
        method_result = acc_change_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
        print('method:', method)
        print(*np.round(np.mean(method_result,axis=0),decimals=3),sep=',')

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    # parser.add_argument('-c','--new_img_num',help='number of new images',type=int)
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    # parser.add_argument('-b','--batches',type=int,default=1)
    # parser.add_argument('-s','--segments',type=int,default=1)
    # parser.add_argument('-bt','--bottom',type=bool,default=True)
    # parser.add_argument('-s','--stride',type=int,default=0)


    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir)
    # print(args.method)