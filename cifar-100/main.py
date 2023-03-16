from utils.strategy import *
from utils import config

def run(ds, old_model_config, new_model_config, acquisition_config, methods, new_img_num_list):
    subset_length_record = {
        k: len(ds[k]) for k in ds.keys() 
    }

    percent_epoch = []

    for method in methods:
        print('In method', method)
        for new_img_num in new_img_num_list:
            acquisition_config.set_items(method,new_img_num)
            if method not in ['seq','seq_clf']:
                new_model = non_seq(ds, old_model_config, new_model_config, acquisition_config)
            elif method == 'seq':
                new_model, minority_cnt = seq(ds, old_model_config, new_model_config, acquisition_config)
                percent_epoch.append(minority_cnt/new_img_num)
            else:
                new_model, minority_cnt = seq_clf(ds, old_model_config, new_model_config, acquisition_config)
                percent_epoch.append(minority_cnt/new_img_num)

            new_model_config.set_path(acquisition_config)
            save_model(new_model,new_model_config.path)

            del new_model

            for split in ds.keys():
                assert subset_length_record[split] == len(ds[split]), 'init {} is changed'.format(split)

    return percent_epoch

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', methods='', seq_rounds=1):
    print('Use pure: ',pure)
    hparams = config['hparams']
    batch_size = hparams['batch_size'][model_dir]

    data_config = config['data']
    select_fine_labels = data_config['selected_labels'][model_dir]
    label_map = data_config['label_map'][model_dir]

    percent_list = []
    superclass_num = int(model_dir.split('-')[0])
    method_list = ['dv','sm','conf','mix'] if methods=='non_seq' else [methods]
    new_img_num_list = data_config['acquired_num_per_class']['mini'] if 'mini' in model_dir else data_config['acquired_num_per_class']['non-mini']

    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = create_dataset_split(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            modify_coarse_label(ds,label_map)

        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        new_model_config = NewModelConfig(batch_size,superclass_num,model_dir,pure, new_model_setter)
        acquistion_config = AcquistionConfig(model_cnt=epo, sequential_rounds= seq_rounds)

        percent_epoch = run(ds,old_model_config,new_model_config,acquistion_config, method_list, new_img_num_list)
        # acc_change_list.append(acc_change_epoch)
        percent_list.append(percent_epoch)

    # print('acc change')
    # acc_change_list = np.round(acc_change_list,decimals=3)
    # # print(*acc_change_list.tolist(),sep='\n')
    # print('acc change average')
    # for m_idx,method in enumerate(method_list): 
    #     method_result = acc_change_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:method_list
    #     print('method:', method)
    #     print(*np.round(np.mean(method_result,axis=0),decimals=3),sep=',')
    percent_list = np.array(percent_list)
    print(np.round(np.mean(percent_list,axis=0),decimals=3))
    print(data_config)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')
    parser.add_argument('-r','--rounds',type=int,default=2)
    parser.add_argument('-m','--methods',type=str)

    args = parser.parse_args()
    # method, new_img_num, save_model
    main(args.epochs,pure=args.pure,model_dir=args.model_dir,seq_rounds=args.rounds,methods=args.methods)
    # print(args.method)