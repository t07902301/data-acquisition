from src.utils import *
def non_seq(method, market_gt, market_preds, market_conf, dv, new_img_num, pure, new_model_setter, subset_loader, ds, base_model, cls_num, batch_size):
    org_train_ds_len = len(ds['train'])
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
    ds['train'] = new_img_set if pure else torch.utils.data.ConcatDataset([ds['train'],new_img_set])

    if method not in ['hard','easy']:
        if pure:
            assert len(ds['train']) == new_img_num*cls_num
        else:
            assert len(ds['train']) == org_train_ds_len + new_img_num*cls_num

    new_train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True,drop_last=True)

    if new_model_setter=='refine':
        new_model = tune(base_model,new_train_loader,subset_loader['val']) # tune
    else:
        new_model = train_model(new_train_loader,subset_loader['val'],num_class=cls_num) # retrain
    return new_model

def seq_clf(ds, base_model, subset_loader, rounds, new_img_num, cls_num, batch_size, pure):
    org_val_ds = ds['val']

    try:
        clf,clip_features,_ = get_CLF(base_model,subset_loader)
    except:
        return -100

    minority_cnt = 0
    new_img_total = None

    for round_i in range(rounds):

        # Eval SVM
        market_gt,market_preds,_ = evaluate_model(subset_loader['market'],base_model)

        # Get new data
        _, dv, _ = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)
        org_market_ds_len = len(ds['market'])
        org_train_ds_len = len(ds['train'])
        org_val_ds_len = len(ds['val'])
        assert len(ds['market'])==len(ds['market_aug']), "market set size not equal to aug market in round {}".format(round_i)

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
        new_img_round = torch.utils.data.Subset(ds['market_aug'],new_img_indices_round)

        mask_kept_img = np.ones(org_market_ds_len,dtype=bool)
        mask_kept_img[new_img_indices_round] = False
        ds['market'] = torch.utils.data.Subset(ds['market'],np.arange(org_market_ds_len)[mask_kept_img])
        ds['market_aug'] = torch.utils.data.Subset(ds['market_aug'],np.arange(org_market_ds_len)[mask_kept_img])
        market_loader = torch.utils.data.DataLoader(ds['market'], batch_size=batch_size)
        subset_loader['market'] = market_loader      
        # print(len(ds['market']), org_market_ds_len,new_img_per_round,cls_num)   
        assert len(ds['market']) == org_market_ds_len - new_img_per_round*cls_num, "error with new market size in round {}".format(round_i)

        ds['val'] = torch.utils.data.ConcatDataset([ds['val'],new_img_round])
        val_loader = torch.utils.data.DataLoader(ds['val'], batch_size=batch_size)
        subset_loader['val'] = val_loader      
        assert len(ds['val']) == org_val_ds_len + new_img_per_round*cls_num, "error with new val size in round {}".format(round_i)   

        # update SVM
        clf,clip_features,_ = get_CLF(base_model,subset_loader)
        minority_cnt += minority_in_ds(new_img_round)
        new_img_total = new_img_round  if (new_img_total == None) else  torch.utils.data.ConcatDataset([new_img_total,new_img_round])
    
    assert len(new_img_total) == new_img_per_round*cls_num*rounds, "error with new data size"
    if pure:
        new_train_set = new_img_total
    else:
        new_train_set = torch.utils.data.ConcatDataset([ds['train'],new_img_total])
        assert len(new_train_set) == org_train_ds_len + new_img_per_round*cls_num*rounds, "error with new train size"  
   
    new_train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    org_val_loader = torch.utils.data.DataLoader(org_val_ds, batch_size=batch_size)
    assert len(org_val_ds) == len(ds['val']) - new_img_per_round*cls_num*rounds, "error with original val size"

    model = train_model(new_train_loader,org_val_loader)    
    return model

def seq(base_model, ds, new_img_num, rounds, subset_loader, cls_num, batch_size, new_model_setter, pure):
    model = base_model
    new_img_per_round = new_img_num//rounds

    minority_cnt = 0
    new_img_total = None
    for round_i in range(rounds):

        # Get SVM 
        clf,clip_features,_ = get_CLF(model,subset_loader) #CLF: a metric of model and data

        # Eval SVM
        market_gt,market_preds,_ = evaluate_model(subset_loader['market'],model)

        # Get decision values
        _, dv, _ = clf.predict(gts=market_gt, latents=clip_features['market'], compute_metrics=False, preds=market_preds)
    
        org_market_ds_len = len(ds['market'])
        org_train_ds_len = len(ds['train'])
        new_img_indices_round = []
        assert len(ds['market'])==len(ds['market_aug']), "market set size not equal to aug market in round {}".format(round_i)
        for c in range(cls_num):
            # Get new data
            mask = market_gt==c
            masked_indices = np.arange(len(mask))[mask]
            masked_dv = dv[mask]

            sorted_masked_dv_idx = np.argsort(masked_dv)
            new_img_indices = top_dv(masked_indices[sorted_masked_dv_idx],new_img_per_round,clf=config['clf'])
            new_img_indices_round.append(new_img_indices)

        new_img_indices_round = np.concatenate(new_img_indices_round)
        new_img_round = torch.utils.data.Subset(ds['market_aug'],new_img_indices_round)

        ds['train'] = torch.utils.data.ConcatDataset([ds['train'],new_img_round]) 
        train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True,drop_last=True)
        subset_loader['train'] = train_loader
        assert len(ds['train']) == org_train_ds_len + new_img_per_round*cls_num, "error with the size of new train set in round {}".format(round_i)   

        # Exclude new images from the market
        mask_kept_img = np.ones(org_market_ds_len,dtype=bool)
        mask_kept_img[new_img_indices_round] = False
        ds['market'] = torch.utils.data.Subset(ds['market'],np.arange(org_market_ds_len)[mask_kept_img])
        ds['market_aug'] = torch.utils.data.Subset(ds['market_aug'],np.arange(org_market_ds_len)[mask_kept_img])
        market_loader = torch.utils.data.DataLoader(ds['market'], batch_size=batch_size)
        subset_loader['market'] = market_loader   
        # print(len(ds['market']), org_market_ds_len,new_img_per_round,cls_num)   
        assert len(ds['market']) == org_market_ds_len - new_img_per_round*cls_num, "error with the size of new market set in round {}".format(round_i)

        if new_model_setter == 'retrain':
            model = train_model(subset_loader['train'],subset_loader['val'],num_class=cls_num)
        else:
            model = tune(model,subset_loader['train'],subset_loader['val'])
        
        minority_cnt += minority_in_ds(new_img_round)
        new_img_total = new_img_round  if (new_img_total == None) else  torch.utils.data.ConcatDataset([new_img_total,new_img_round])
   
    assert len(new_img_total) == new_img_per_round*cls_num*rounds, 'error with the number of new data'
    if pure:
        new_train_loader = torch.utils.data.DataLoader(new_img_total, batch_size=batch_size, shuffle=True,drop_last=True)
        model = train_model(new_train_loader,subset_loader['val'],num_class=cls_num)
    
    return model, minority_cnt   