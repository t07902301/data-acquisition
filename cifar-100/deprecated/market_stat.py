from utils import *

def run(cls_num,ds,methods,new_img_num_list,epoch=0,new_model_setter='refine'):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], 
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

    percent = []

    new_model_path_root = os.path.join(model_path_root,new_model_setter)
    if os.path.exists(new_model_path_root) is False:
        os.makedirs(new_model_path_root)

    for method in methods:
        print('In method', method)
        percent_method = []

        for new_img_num in new_img_num_list:
            model = deepcopy(base_model)
            train_ds = ds['train']
            org_train_ds_len = len(train_ds)
            new_img_indices_total = []

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
                new_img_indices_total.append(new_img_indices)
            
            new_img_indices_total =  np.concatenate(new_img_indices_total)
            new_img = torch.utils.data.Subset(ds['market'],new_img_indices_total)
            cnt = minority_in_ds(new_img)
            percent_method.append(cnt/len(new_img))

        percent.append(percent_method)
    return percent

def main(epochs,new_img_num,metric=''):
    percent_list = []
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)

    # select_fine_labels = [30, 1, 62, 9, 0, 22, 5, 6, 42, 17, 23, 15, 34, 26, 11, 27, 36, 47, 8, 41, 4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    # select_fine_labels = None
    select_fine_labels = [30, 4, 9, 10, 0, 51]
    sp_labels_map = {
        0: 0,
        3: 1,
        4: 2
    }

    new_img_num_list = [25,50,75,100]
    # new_img_num_list = [75]
  
    method_list = ['dv','sm','conf','mix']
    # method_list = ['dv']
    # method_list = ['hard', 'easy','dv','sm','conf','mix']
    # method_list = ['hard', 'easy']
   
    percent_list = []

    cls_num = 20 if select_fine_labels==None else len(select_fine_labels)//2

    for i in range(epochs):

        print('in epoch {}'.format(i))

        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
        
        for split,infos in ds.items():
            new_ds_split = []
            for info in infos:
                new_ds_split.append((info[0],sp_labels_map[info[1]],info[2]))
            ds[split] = new_ds_split 

        percent = run(cls_num=cls_num,methods=method_list,new_img_num_list=new_img_num_list,ds=ds,epoch=i,new_model_setter='retrain')
        percent_list.append(percent)
    print('percent ')
    percent_list = np.round(percent_list,decimals=3)
    for m_idx,method in enumerate(method_list): 
        method_result = percent_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
        print('method:', method)
        print(*np.round(np.mean(method_result,axis=0),decimals=3),sep=',') #(e,img_num)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        # help='an integer for the accumulator')
    parser.add_argument('-c','--new_img_num',help='number of new images',type=int)
    parser.add_argument('-e','--epochs',type=int,default=1)

    # parser.add_argument('-b','--batches',type=int,default=1)
    # parser.add_argument('-s','--segments',type=int,default=1)
    # parser.add_argument('-bt','--bottom',type=bool,default=True)
    # parser.add_argument('-s','--stride',type=int,default=0)


    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,args.new_img_num)
    # print(args.method)