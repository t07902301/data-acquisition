from utils.strategy import *
from utils.set_up import set_up
def run(ds:Dataset.DataSplits, model_config:Config.OldModel):
    remove_fine_labels = config['data']['remove_fine_labels']    
    base_model = Model.load(model_config)

    test_fine_labels = Dataset.get_ds_labels(ds.dataset['test_shift'])
    test_shift_indices = np.arange(len(ds.dataset['test_shift']))
    removed_indices = []
    for label in remove_fine_labels:
        removed_mask = (test_fine_labels==label)
        removed_indices.append(test_shift_indices[removed_mask])
    removed_indices = np.concatenate(removed_indices)
    keep_mask = np.ones(len(test_shift_indices),dtype=bool)
    keep_mask[removed_indices] = False
    keep_indices = test_shift_indices[keep_mask]
    assert len(removed_indices) + len(keep_indices) == len(test_shift_indices)

    removed_test_shift = torch.utils.data.Subset(ds.dataset['test_shift'], removed_indices)
    ds.replace('new_model', removed_test_shift)
    remained_test_shift = torch.utils.data.Subset(ds.dataset['test_shift'], keep_indices)
    ds.replace('old_model', remained_test_shift)

    market_fine_labels = Dataset.get_ds_labels(ds.dataset['market'])
    market_indices = np.arange(len(ds.dataset['market']))
    removed_indices = []
    for label in remove_fine_labels:
        removed_indices.append(market_indices[market_fine_labels==label])
    removed_indices = np.concatenate(removed_indices)
    removed_market = torch.utils.data.Subset(ds.dataset['market'], removed_indices)
    ds.replace('train', removed_market)
    m1 = Model.train(ds.loader['train'], ds.loader['val_shift'], num_class=2)

    gt,pred,_  = Model.evaluate(ds.loader['new_model'],m1)
    m1_new_acc = (gt==pred)
    gt,pred,_  = Model.evaluate(ds.loader['old_model'],base_model)
    m1_old_acc = (gt==pred)
    m1_total = np.concatenate((m1_new_acc,m1_old_acc))
    m1_acc = m1_total.mean()*100

    val_fine_labels = Dataset.get_ds_labels(ds.dataset['val_shift'])
    val_indices = np.arange(len(ds.dataset['val_shift']))
    removed_indices = []
    for label in remove_fine_labels:
        removed_indices.append(val_indices[val_fine_labels==label])
    removed_indices = np.concatenate(removed_indices)
    removed_val = torch.utils.data.Subset(ds.dataset['val_shift'], removed_indices)
    ds.replace('new_val_shift', removed_val)
    m2 = Model.train(ds.loader['train'], ds.loader['new_val_shift'], num_class=2)
    gt,pred,_  = Model.evaluate(ds.loader['new_model'],m2)
    m2_new_acc = (gt==pred)
    gt,pred,_  = Model.evaluate(ds.loader['old_model'],base_model)
    m2_old_acc = (gt==pred)
    m2_total = np.concatenate((m2_new_acc,m2_old_acc))
    m2_acc = m2_total.mean()*100

    gt,pred,_  = Model.evaluate(ds.loader['test_shift'],base_model)
    base_acc = (gt==pred).mean()*100
    
    return m1_acc-base_acc, m2_acc-base_acc

def main(epochs,  model_dir ='', device=0):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    m1_acc, m2_acc = [], []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        m1, m2 = run(ds, old_model_config)
        m1_acc.append(m1)
        m2_acc.append(m2)
    print('Model 1 Average Acc after shift:', np.round(np.mean(m1_acc),decimals=3))
    print('Model 2 Average Acc after shift:', np.round(np.mean(m2_acc),decimals=3))
    
    split_data = ds.dataset
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir,args.device)