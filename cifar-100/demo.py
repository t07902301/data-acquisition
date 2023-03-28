from utils.strategy import *
from utils.statistics import get_threshold_collection
import matplotlib.pyplot as plt
def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', methods='', seq_rounds=1):
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, pure)
    new_img_num_list = [150]
    for epo in range(epochs):
        n_cols = len(new_img_num_list)
        n_rows = 2
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, tight_layout=True)
        axs = axs.flatten()
        print('in epoch {}'.format(epo))

        ds = DataSplits(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            ds.modify_coarse_label(label_map)
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        # new_model_config = NewModelConfig(batch_size,superclass_num,model_dir,pure, new_model_setter)
        acquistion_config = AcquistionConfigFactory(methods,seq_rounds)
        idx_log_config = LogConfig(batch_size,superclass_num,model_dir, epo, pure, new_model_setter)
        idx_log_config.root = os.path.join(idx_log_config.root, 'indices')
        
        base_model = load_model(old_model_config.path)

        for col in range(n_cols):
            ds.get_dataloader(batch_size)
            clf,clip_processor,_ = get_CLF(base_model,ds.loader)

            market_info = apply_CLF(clf, ds.loader['market'], clip_processor, base_model)
            market_info_aug = apply_CLF(clf, ds.loader['market_aug'], clip_processor, base_model)        
            n_data = new_img_num_list[col]
            acquistion_config.set_items('dv',n_data)
            idx_log_config.set_path(acquistion_config)
            indices = load_log(idx_log_config.path)
            indices = np.concatenate(indices)
            axs[col].hist(market_info['dv'][indices], bins=10)
            axs[col+n_cols].hist(market_info_aug['dv'][indices], bins=10)

        fig.savefig('figure/dv_aug_org_market/{}.png'.format(epo))
        fig.clf()

def main(epochs,model_dir,pure=False,new_model_setter='retrain'):
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio = parse_config(model_dir, pure)
    ds_list = get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    n_data_list = [200]
    dv_list = [[],[]]
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir, epo, pure, new_model_setter)
        acquistion_config = AcquistionConfig()
        idx_log_config = LogConfig(batch_size=new_model_config.batch_size,class_number=new_model_config.class_number,model_dir=new_model_config.model_dir,pure=new_model_config.pure,setter=new_model_config.setter,model_cnt=epo)
        idx_log_config.root = os.path.join(idx_log_config.root, 'indices')
        threshold_collection = get_threshold_collection(n_data_list, acquistion_config, new_model_config, old_model_config, ds, epo)

        base_model = load_model(old_model_config.path)
        clf,clip_processor,_ = get_CLF(base_model,ds.loader)
        market_info = apply_CLF(clf,ds.loader['market'],clip_processor)
        for n_data in n_data_list:
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
        for n_data in n_data_list:
            threshold = threshold_collection[n_data]
            n_class = len(threshold)
            selected_indices_total = []
            for c in range(n_class):
                cls_indices, cls_mask, cls_dv = get_class_info(c, test_info['gt'], test_info['dv'])
                dv_selected_indices = (cls_dv<=threshold[c])
                selected_indices_total.append(dv_selected_indices)
            dv_list[1].append( test_info['dv'][np.concatenate(selected_indices_total)])

    n_cols = len(n_data_list)*epochs
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, tight_layout=True)
    axs = axs.flatten()
    for col in range(n_cols):
        axs[col].hist(dv_list[0][col], bins=6)
        axs[col+n_cols].hist(dv_list[1][col], bins=6)  
    fig.savefig('figure/3-class-mini-small/selected_market_test.png')
    fig.clf()      
main(epochs=5, model_dir='3-class-mini-small', pure=True)