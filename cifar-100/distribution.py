from utils.strategy import *
from utils.statistics import get_threshold_collection
import matplotlib.pyplot as plt

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0, n_data=150):
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio = parse_config(model_dir, pure)
    ds_list = get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    dv_list = [[],[], []]
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    for epo in range(epochs):
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter)
        acquistion_config = AcquistionConfig()
        threshold_collection = get_threshold_collection([n_data], acquistion_config, new_model_config, old_model_config, ds, epo)
        base_model = load_model(old_model_config)
        clf,clip_processor,_ = get_CLF(base_model,ds.loader)

        market_info = apply_CLF(clf,ds.loader['market'],clip_processor)
        threshold = threshold_collection[n_data]
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices, cls_mask, cls_dv = get_class_info(c, market_info['gt'], market_info['dv'])
            dv_selected_indices = (cls_dv<=threshold[c])
            selected_indices_total.append(dv_selected_indices)
            # assert dv_selected_indices.sum() == n_data, dv_selected_indices.sum()
        dv_list[0].append( market_info['dv'][np.concatenate(selected_indices_total)])         

        test_info = apply_CLF(clf,ds.loader['test'],clip_processor)
        threshold = threshold_collection[n_data]
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices, cls_mask, cls_dv = get_class_info(c, test_info['gt'], test_info['dv'])
            dv_selected_indices = (cls_dv<=threshold[c])
            selected_indices_total.append(dv_selected_indices)
        dv_list[1].append( test_info['dv'][np.concatenate(selected_indices_total)])

        market_info_aug = apply_CLF(clf,ds.loader['market_aug'],clip_processor)
        threshold = threshold_collection[n_data]
        n_class = len(threshold)
        selected_indices_total = []
        for c in range(n_class):
            cls_indices, cls_mask, cls_dv = get_class_info(c, market_info_aug['gt'], market_info_aug['dv'])
            dv_selected_indices = (cls_dv<=threshold[c])
            selected_indices_total.append(dv_selected_indices)
        dv_list[2].append( market_info_aug['dv'][np.concatenate(selected_indices_total)])    

    n_cols = epochs
    n_rows = len(dv_list)
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, tight_layout=True)
    axs = axs.flatten()
    for col in range(n_cols):
        axs[col].hist(dv_list[0][col], bins=6)
        axs[col+n_cols].hist(dv_list[1][col], bins=6)  
        axs[col+2*n_cols].hist(dv_list[2][col], bins=6)  
    fig.savefig('figure/{}/ds_dtr_{}.png'.format(model_dir,n_data))
    fig.clf()      
    print('save fig to', 'figure/{}/ds_dtr_{}.png'.format(model_dir,n_data))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-nd','--n_data',type=int,default=0)
    parser.add_argument('-p','--pure',type=bool,default=True)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,model_dir=args.model_dir,device=args.device, pure=args.pure, n_data=args.n_data)