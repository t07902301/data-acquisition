from random import shuffle
from utils import *

# Load subsets
train_ds,test_ds = load_dataset()
print('data subsets loaded')
org_train_ds_indices = np.arange(len(train_ds))

bsz = hparams['batch_size']
model_path_root = os.path.join(config["model_path_root"],'_'.join(selected_classes),config['optimizer'])
train_size = config["train_size"]
val_size = config["val_size"]
market_size = config["market_size"]

if os.path.exists(model_path_root) is False:
    os.makedirs(model_path_root)

def save_model(model, path):
    torch.save({
        'build_params': model._build_params,
        'state_dict': model.state_dict(),
    }, path)
def run(cls_num=2,ds = None,model_cnt=0,train_flag=False,metric=''):
    aug_train_ds,org_train_ds,val_ds,market_ds = ds
    train_loader = torch.utils.data.DataLoader(aug_train_ds, batch_size=bsz, shuffle=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bsz) # loader = batches of points
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bsz)
    no_aug_train_loader = torch.utils.data.DataLoader(org_train_ds, batch_size=bsz)
    market_loader = torch.utils.data.DataLoader(market_ds, batch_size=bsz)

    subset_loader = {'train': no_aug_train_loader, 'test': test_loader, 'val': val_loader, 'market': market_loader} 

    path = os.path.join(model_path_root,'{}.pt'.format(model_cnt))
    base_model = get_base_model(train_loader,subset_loader['val'],model_save_path=path,train_flag=train_flag)

    if train_flag:
        # Get a new base model from Resnet
        save_model(base_model, path)
        print('base model saved to {}'.format(path))
    else:
        # Load base model
        print('base model loaded from {}'.format(path))

    # # Evaluate overall
    # base_model = base_model.eval()
    # gt,pred,loss  = evaluate_model(subset_loader['test'],base_model)
    # base_acc = np.sum(np.where(gt==pred,1,0))/gt.shape*100
    # base_loss = torch.mean(loss)

    # Evaluate
    base_model = base_model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
    base_acc = get_metrics(gt,pred,metric, confusion_matrix=True)
    gt,pred,_  = evaluate_model(subset_loader['val'],base_model)
    val_acc = get_metrics(gt,pred,metric)

    # Get SVM 
    clf,clip_features,clf_score = get_CLF(base_model,subset_loader,metric=metric)

    # Evaluate SVM on the market set
    market_gt,market_preds,_  = evaluate_model(subset_loader['market'],base_model)
    market_acc = get_metrics(market_gt,market_preds,metric, confusion_matrix=True)
    _, dv, clf_eval_score = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)
    clf_score['eval'] = clf_eval_score

    model_acc = {'clf_fit':val_acc, 'clf_eval':market_acc}
    # fid = []
    # for i in range(cls_num):
    #     fid.append(calculate_fid_given_paths(val_clf[i],market_clf[i]))
    
    return base_acc,clf_score, model_acc

def main(epochs,train_flag,new_market_use=False,metric=''):
    base_acc_list = []
    fid_list = []
    cv_scores_list = []
    val_score_list, market_score_list = [], []
    val_acc_list, market_acc_list = [],[]    

    # train_val_indices = get_subsets_indices(org_train_ds_indices,epochs,subset_size=train_size+val_size+market_size)

    if new_market_use:
        new_market = load_market('/home/yiwei/data/cars/train/',0)

    for i in range(epochs):
  
        # # aug_train_ds,org_train_ds,val_ds = get_data_split(split_indicies[i],train_ds)
        # train_val_subset = torch.utils.data.Subset(train_ds,train_val_indices[i])
        # org_train_ds,val_ds = torch.utils.data.random_split(train_val_subset,[train_size,val_size+market_size],generator=generator) # np.random won't affect this random number generator
        # val_ds,market_ds = torch.utils.data.random_split(val_ds,[val_size,market_size],generator=generator)

        train_dataset = torch.utils.data.Subset(train_ds,np.arange(len(train_ds)))
        org_train_ds,val_market_ds = torch.utils.data.random_split(train_dataset,[train_size,val_size+market_size],generator=generator) # np.random won't affect this random number generator
        val_ds,market_ds = torch.utils.data.random_split(val_market_ds,[val_size,market_size],generator=generator)

        print(val_ds[0][0].shape)
        print(test_ds[0][0].shape)

        # aug_train_ds = copy.deepcopy(org_train_ds)
        # aug_train_ds.dataset.transform = train_transform

        aug_train_ds = org_train_ds

        if new_market_use:
            market_ds = new_market
        
        img_list_1,img_list_2 = ds_to_list(train_ds,test_ds)
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
        fid_value = calculate_fid_given_paths([img_list_1,img_list_2],
                                            batch_size=50,
                                            device=device,
                                            dims=2048,
                                            num_workers=num_workers)       
        # print(fid_value)
        fid_list.append(fid_value)


    print(config['selected_class'],config['clf'],config['clf_args'],train_size,val_size,market_size)     
    print('fid:')
    print(np.round(fid_list,decimals=3))
    # print(np.round(np.mean(fid_list),decimals=3))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-m', '--metric',type=str,default='precision')

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.train_flag,metric=args.metric)
    # print(args.method)