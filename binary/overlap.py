from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.stat_test as stat_test
old_labels = set(Dataset.data_config['train_label']) - set(Dataset.data_config['remove_fine_labels'])
def remove_old_data(dataset, sample_ratio):
    _, left_data = Dataset.split_dataset(dataset, old_labels, sample_ratio)
    return left_data

def run(ds:Dataset.DataSplits, model_config:Config.OldModel, clip_processor, old_data_ratio):
    if model_config.base_type == 'svm':
        base_model = Model.svm(ds.loader['train_clip'], clip_processor)
    else:
        base_model = Model.resnet(2)
    base_model.load(model_config)

    # Evaluate
    gt,pred,_  = base_model.eval(ds.loader['test'])
    acc = (gt==pred).mean()*100
    gt,pred,_  = base_model.eval(ds.loader['test_shift'])
    acc_shift = (gt==pred).mean()*100

    clf = Detector.SVM(ds.loader['train_clip'], clip_processor)
    score = clf.fit(base_model, ds.loader['test_shift'])
    _, detect_prec = clf.predict(ds.loader['test_shift'], compute_metrics=True, base_model=base_model)

    shift_score = Subset.mis_cls_stat('test_shift', ds, base_model)

    intersection_area = stat_test.run(clf, ds.loader['test_shift'], base_model, model_config, old_data_ratio)

    return acc, acc_shift, detect_prec, shift_score, intersection_area

def ratio_run(ratio, dataset, old_model_config, clip_processor):
    ds = deepcopy(dataset)
    ds['test_shift'] = remove_old_data(ds['test_shift'], ratio)
    ds = Dataset.DataSplits(ds, old_model_config.batch_size)
    acc, acc_shift, detect_prec, shift_score, intersection_area = run(ds, old_model_config, clip_processor, ratio)
    print(len(ds.dataset['test_shift']))
    return acc, acc_shift, detect_prec, shift_score, intersection_area.tolist()[0]*100


def epoch_run(old_model_config: Config.OldModel, dataset, old_data_ratio_list, clip_processor):
    acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo = [], [], [], [], []
    for old_data_ratio in old_data_ratio_list:
        acc, acc_shift, detect_prec, shift_score, intersection_area = ratio_run(old_data_ratio, dataset, old_model_config, clip_processor)
        acc_epo.append(acc)
        acc_shift_epo.append(acc_shift)
        detect_prec_epo.append(detect_prec)
        shift_epo.append(shift_score)
        intersection_area_epo.append(intersection_area)
    return acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo

def main(epochs,  model_dir ='', device_id=0, base_type=''):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    clip_processor = Detector.load_clip(device_config)

    old_data_ratio_list, _ = np.linspace(0, 1, retstep=True, num=5)
    acc_list, acc_shift_list, detect_prec_list, shift_list, intersection_area_list = [], [], [], [], []

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, base_type)
        ds_init = ds_list[epo]
        acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo = epoch_run(old_model_config, ds_init, old_data_ratio_list, clip_processor)
        acc_list.append(acc_epo)
        acc_shift_list.append(acc_shift_epo)
        detect_prec_list.append(detect_prec_epo)
        shift_list.append(shift_epo)
        intersection_area_list.append(intersection_area_epo)
    # Print the table headers
    print('{}(%), {}(%), {}(%), {}(%), {}(%),'.format('Old Data Removal from Test Set', 'Base Model Average Acc', 'SVM Precision', 'New Data Percent on Model Mistakes', 'Overlapped Area'))

    acc_shift_list = np.round(np.mean(acc_shift_list, axis=0), decimals=3)
    detect_prec_list = np.round(np.mean(detect_prec_list, axis=0), decimals=3)
    shift_list = np.round(np.mean(shift_list, axis=0), decimals=3)
    intersection_area_list = np.round(np.mean(intersection_area_list, axis=0), decimals=3)

    # Print the table rows
    for i in range(len(old_data_ratio_list)):
        print('{}, {}, {}, {}, {} '.format(old_data_ratio_list[i]*100, acc_shift_list[i], detect_prec_list[i], shift_list[i], intersection_area_list[i]))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='resnet')

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir, args.device, args.base_type)