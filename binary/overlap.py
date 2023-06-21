from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.stat_test as stat_test
old_labels = set(Dataset.data_config['train_label']) - set(Dataset.data_config['remove_fine_labels'])
def remove_old_data(dataset, sample_ratio):
    np.random.seed(1)
    _, left_data = Dataset.split_dataset(dataset, old_labels, sample_ratio)
    return left_data

def run(ds:Dataset.DataSplits, model_config:Config.OldModel, clip_processor, old_data_ratio, clf_fit_ds, clf_check_ds):
    if model_config.base_type == 'svm':
        base_model = Model.svm(ds.loader['train_clip'], clip_processor)
    else:
        base_model = Model.resnet(2)
    base_model.load(model_config)

    gt,pred,_  = base_model.eval(ds.loader[clf_fit_ds])
    acc_shift = (gt==pred).mean()*100

    clf = Detector.resnet(2)
    clf.fit(base_model, ds.dataset[clf_fit_ds], ds.loader[clf_fit_ds], model_config.batch_size)
    _, clf_acc = clf.predict(ds.loader[clf_fit_ds], base_model, model_config.batch_size, compute_metrics=True)
    _, detect_prec = clf.predict(ds.loader[clf_check_ds], base_model, model_config.batch_size, compute_metrics=True)
    intersection_area = stat_test.run(clf, ds.loader[clf_check_ds], model_config, old_data_ratio, plot=True, overlap_cut_value=0.5)

    # clf = Detector.SVM(ds.loader['train_clip'], clip_processor, split_and_search=True, data_transform='flatten')
    # _ = clf.fit(base_model, ds.loader[clf_fit_ds])
    # print('In CLF fit domain:')
    # _, clf_acc = clf.predict(ds.loader[clf_fit_ds], base_model, compute_metrics=True)
    # _, detect_prec = clf.predict(ds.loader[clf_check_ds], base_model, compute_metrics=True)
    # intersection_area = stat_test.run(clf, ds.loader[clf_check_ds], model_config, old_data_ratio, plot=True, overlap_cut_value=0)

    shift_score = Subset.mis_cls_stat(clf_fit_ds, ds, base_model)

    # intersection_area = stat_test.run(clf, ds.loader[clf_check_ds], model_config, old_data_ratio, plot=True, overlap_cut_value=0.5)

    return acc_shift, detect_prec, shift_score, intersection_area, clf_acc

def ratio_run(ratio, dataset, old_model_config, clip_processor, clf_fit_ds = 'val_shift', clf_check_ds='test_shift'):
    ds = deepcopy(dataset)
    ds[clf_fit_ds] = remove_old_data(ds[clf_fit_ds], ratio)
    print(clf_fit_ds, len(ds[clf_fit_ds]))
    print(clf_check_ds, len(ds[clf_check_ds]))

    ds = Dataset.DataSplits(ds, old_model_config.batch_size)
    acc_shift, detect_prec, shift_score, intersection_area, clf_acc = run(ds, old_model_config, clip_processor, ratio, clf_fit_ds, clf_check_ds)
    return acc_shift, detect_prec, shift_score, intersection_area.tolist()[0]*100, clf_acc

def epoch_run(old_model_config: Config.OldModel, dataset, old_data_ratio_list, clip_processor):
    acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo, clf_acc_epo = [], [], [], [], [], []
    for old_data_ratio in old_data_ratio_list:
        acc_shift, detect_prec, shift_score, intersection_area, clf_acc = ratio_run(old_data_ratio, dataset, old_model_config, clip_processor)
        acc_shift_epo.append(acc_shift)
        detect_prec_epo.append(detect_prec)
        shift_epo.append(shift_score)
        intersection_area_epo.append(intersection_area)
        clf_acc_epo.append(clf_acc)
    return acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo, clf_acc_epo

def main(epochs,  model_dir ='', device_id=0, base_type=''):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    clip_processor = Detector.load_clip(device_config)

    old_data_ratio_list, _ = np.linspace(0, 1, retstep=True, num=5)
    # old_data_ratio_list = [0]
    acc_list, acc_shift_list, detect_prec_list, shift_list, intersection_area_list, clf_acc_list = [], [], [], [], [], []

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, base_type)
        ds_init = ds_list[epo]
        acc_epo, acc_shift_epo, detect_prec_epo, shift_epo, intersection_area_epo, clf_acc = epoch_run(old_model_config, ds_init, old_data_ratio_list, clip_processor)
        acc_list.append(acc_epo)
        acc_shift_list.append(acc_shift_epo)
        detect_prec_list.append(detect_prec_epo)
        shift_list.append(shift_epo)
        intersection_area_list.append(intersection_area_epo)
        clf_acc_list.append(clf_acc)
        
    # Print the table headers
    print('{}(%), {}(%), {}(%), {}(%), {}(%),'.format('Old Data Removal Ratio', 'Base Model Average Acc', 'New Data Percent on Model Mistakes', 'SVM Balanced Acc', 'Overlapped Area'))

    acc_shift_list = np.round(np.mean(acc_shift_list, axis=0), decimals=3)
    detect_prec_list = np.round(np.mean(detect_prec_list, axis=0), decimals=3)
    shift_list = np.round(np.mean(shift_list, axis=0), decimals=3)
    intersection_area_list = np.round(np.mean(intersection_area_list, axis=0), decimals=3)
    clf_acc_list = np.round(np.mean(clf_acc_list, axis=0), decimals=3)

    # Print the table rows
    for i in range(len(old_data_ratio_list)):
        print('{}, {}, {}, {}, {}, {} '.format(old_data_ratio_list[i]*100, acc_shift_list[i], shift_list[i], clf_acc_list[i], detect_prec_list[i], intersection_area_list[i]))

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