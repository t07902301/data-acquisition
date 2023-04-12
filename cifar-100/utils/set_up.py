from utils.Config import print_config, parse_config
from utils.dataset import get_data_splits_list
import torch
def set_up(epochs,  model_dir, pure, device=0):
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = parse_config(model_dir, pure)
    print_config(batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio)
    ds_list = get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    return batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config
import numpy as np
def CLF_statistics(epochs, score, n_data_list = None):
    if n_data_list is None:
        cv_score, fit_precision = [], []
        recall, standard = [], []
        for epo in range(epochs):
            cv_score.append(score[epo]['cv'])
            fit_precision.append(score[epo]['fit'])
            # recall.append(score[epo]['recall'])
            # standard.append(score[epo]['standard'])
        print('cv average score:', np.mean(cv_score, axis=0))
        print('fit precision average:', np.mean(fit_precision, axis=0))
        # print('fit recal average:', np.mean(recall, axis=0))
        # print('fit standard average:', np.mean(standard, axis=0))
    else:
        for idx, n_data in enumerate(n_data_list):
            cv_score, fit_precision = [], []
            for epo in range(epochs):
                cv_score.append(score[epo][idx]['cv'])
                fit_precision.append(score[epo][idx]['fit'])     
            print('n_data:', n_data)       
            print('cv average score:', np.round(np.mean(cv_score, axis=0),decimals=3))
            print('fit precision average:', np.round(np.mean(fit_precision, axis=0),decimals=3))            