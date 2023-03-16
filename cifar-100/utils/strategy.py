from utils.dataset import *
from utils.model import *
from utils.acquistion import *
from utils.Config import *
from copy import deepcopy

def non_seq(initial_ds, old_model_config, new_model_config, acquisition_config):
    ds = deepcopy(initial_ds)
    loader = get_dataloader(ds, new_model_config.batch_size)
    pure = new_model_config.pure
    new_data_number_per_class = acquisition_config.new_data_number_per_class
    class_number = old_model_config.class_number

    base_model = load_model(old_model_config.path)
    # Get SVM 
    clf,clip_features,_ = get_CLF(base_model,loader)
    market_info = apply_CLF(clf,base_model,loader,clip_features, 'market')

    org_train_ds_len = len(ds['train'])
    new_data_indices = get_new_data_indices(new_model_config, market_info, acquisition_config.method, new_data_number_per_class)
    new_data_set = torch.utils.data.Subset(ds['market_aug'],new_data_indices)
    ds['train'] = new_data_set if pure else torch.utils.data.ConcatDataset([ds['train'],new_data_set])

    if acquisition_config.method not in ['hard','easy']:
        if pure:
            assert len(ds['train']) == new_data_number_per_class*class_number
        else:
            assert len(ds['train']) == org_train_ds_len + new_data_number_per_class*class_number

    new_train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=new_model_config.batch_size, shuffle=True,drop_last=True)

    if new_model_config.setter=='refine':
        new_model = tune(base_model,new_train_loader,loader['val']) # tune
    else:
        new_model = train_model(new_train_loader,loader['val'],num_class=class_number) # retrain
    return new_model

def seq_clf(initial_ds, old_model_config, new_model_config, acquisition_config):
    ds = deepcopy(initial_ds)
    org_val_ds = ds['val']
    pure = new_model_config.pure
    new_data_number_per_class = acquisition_config.new_data_number_per_class
    class_number = old_model_config.class_number
    minority_cnt = 0
    new_data_total_set = None
    rounds = acquisition_config.sequential_rounds
    new_data_num_round = new_data_number_per_class // rounds
    batch_size = new_model_config.batch_size
    loader = get_dataloader(ds, new_model_config.batch_size)

    model = load_model(old_model_config.path)
    clf,clip_features,_ = get_CLF(model,loader)

    for round_i in range(rounds):

        market_info = apply_CLF(clf,model,loader,clip_features, 'market')
        
        org_market_ds_len = len(ds['market'])
        org_train_ds_len = len(ds['train'])
        org_val_ds_len = len(ds['val'])
        assert len(ds['market'])==len(ds['market_aug']), "market set size not equal to aug market in round {}".format(round_i)

        new_data_round_indices = get_new_data_indices(new_model_config, market_info, acquisition_config.round_acquire_method, new_data_num_round)
        new_data_round_set = torch.utils.data.Subset(ds['market_aug'],new_data_round_indices)
        new_data_round_set_no_aug = torch.utils.data.Subset(ds['market'],new_data_round_indices)
        update_market(ds, new_data_round_indices)

        assert len(ds['market']) == org_market_ds_len - new_data_num_round*class_number, "error with new market size in round {}".format(round_i)

        market_loader = torch.utils.data.DataLoader(ds['market'], batch_size=batch_size)
        loader['market'] = market_loader 

        ds['val'] = torch.utils.data.ConcatDataset([ds['val'],new_data_round_set_no_aug])
        val_loader = torch.utils.data.DataLoader(ds['val'], batch_size=batch_size)
        loader['val'] = val_loader      
        assert len(ds['val']) == org_val_ds_len + new_data_num_round*class_number, "error with new val size in round {}".format(round_i)   

        # update SVM
        clf,clip_features,_ = get_CLF(model,loader)
        minority_cnt += minority_in_ds(new_data_round_set_no_aug)
        new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])
    
    assert len(new_data_total_set) == new_data_num_round*class_number*rounds, "error with new data size"
    if pure:
        new_train_set = new_data_total_set
    else:
        new_train_set = torch.utils.data.ConcatDataset([ds['train'],new_data_total_set])
        assert len(new_train_set) == org_train_ds_len + new_data_num_round*class_number*rounds, "error with new train size"  
   
    new_train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    org_val_loader = torch.utils.data.DataLoader(org_val_ds, batch_size=batch_size)
    assert len(org_val_ds) == len(ds['val']) - new_data_num_round*class_number*rounds, "error with original val size"

    model = train_model(new_train_loader,org_val_loader,num_class=class_number)    
    return model, minority_cnt

def seq(initial_ds, old_model_config, new_model_config, acquisition_config):
    ds = deepcopy(initial_ds)
    pure = new_model_config.pure
    new_data_number_per_class = acquisition_config.new_data_number_per_class
    class_number = old_model_config.class_number
    minority_cnt = 0
    new_data_total_set = None
    rounds = acquisition_config.sequential_rounds
    new_data_num_round = new_data_number_per_class // rounds
    batch_size = new_model_config.batch_size

    loader = get_dataloader(ds, new_model_config.batch_size)
    model = load_model(old_model_config.path)

    for round_i in range(rounds):
        # Get SVM 
        clf,clip_features,_ = get_CLF(model,loader) #CLF: a metric of model and data

        market_info = apply_CLF(clf,model,loader,clip_features, 'market')

        org_market_ds_len = len(ds['market'])
        org_train_ds_len = len(ds['train'])
        assert len(ds['market'])==len(ds['market_aug']), "market set size not equal to aug market in round {}".format(round_i)

        new_data_round_indices = get_new_data_indices(new_model_config, market_info, acquisition_config.round_acquire_method, new_data_num_round)

        new_data_round_set = torch.utils.data.Subset(ds['market_aug'],new_data_round_indices)
        update_market(ds, new_data_round_indices)
        assert len(ds['market']) == org_market_ds_len - new_data_num_round*class_number, "error with new market size in round {}".format(round_i)

        market_loader = torch.utils.data.DataLoader(ds['market'], batch_size=batch_size)
        loader['market'] = market_loader 
        ds['train'] = torch.utils.data.ConcatDataset([ds['train'],new_data_round_set]) 
        train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=batch_size, shuffle=True,drop_last=True)
        loader['train'] = train_loader
        assert len(ds['train']) == org_train_ds_len + new_data_num_round*class_number, "error with the size of new train set in round {}".format(round_i)   

        if new_model_config.setter == 'retrain':
            model = train_model(loader['train'],loader['val'],num_class=class_number)
        else:
            model = tune(model,loader['train'],loader['val'])
        
        minority_cnt += minority_in_ds(new_data_round_set)
        new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])
   
    assert len(new_data_total_set) == new_data_num_round*class_number*rounds, 'error with the number of new data'
    if pure:
        new_train_loader = torch.utils.data.DataLoader(new_data_total_set, batch_size=batch_size, shuffle=True,drop_last=True)
        model = train_model(new_train_loader,loader['val'],num_class=class_number)
    
    return model, minority_cnt   