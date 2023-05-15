from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset

def run(ds:Dataset.DataSplits, model_config:Config.OldModel, train_flag:bool):
    svm = Model.svm(ds.loader['train_clip'])
    if train_flag:
        svm.train(ds.loader['train_clip'])
    else:
        svm.load(model_config)
    # Evaluate
    gts, pred, _ = svm.eval(ds.loader['test'])
    acc = (gts==pred).mean()
    gts, pred, _ = svm.eval(ds.loader['test_shift'])
    acc_shift = (gts==pred).mean()

    dct = Detector.SVM(ds.loader['train_clip'], False) 
    score = dct.fit(svm, ds.loader['val_shift'])
    # print(score)
    _, detect_prec = dct.predict(ds.loader['val_shift'], compute_metrics=True, base_model=svm)  
    shift_score = Subset.shift_importance('val_shift', ds, svm)

    if train_flag:
        svm.save(model_config.path)

    return acc, acc_shift, detect_prec, shift_score

def main(epochs,  model_dir ='', train_flag=False, device_id=0):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    acc_list, acc_shift_list, detect_prec_list, shift_list = [], [], [], []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, 'svm')
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        prec, acc_shift, detect_prec, shift_score = run(ds, old_model_config, train_flag)
        acc_list.append(prec)
        acc_shift_list.append(acc_shift)
        detect_prec_list.append(detect_prec)
        shift_list.append(shift_score)
    print('Model Average Acc before shift:', np.round(np.mean(acc_list),decimals=3))
    print('Model Average Acc after shift:', np.round(np.mean(acc_shift_list),decimals=3))
    Detector.statistics(detect_prec_list, 'precision')
    print('Distribution Shift Proportion on Model Misclassifications: {}%'.format(np.round(np.mean(np.array(shift_list),axis=0), decimals=3)))

    split_data = ds.dataset
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir,args.train_flag, args.device)