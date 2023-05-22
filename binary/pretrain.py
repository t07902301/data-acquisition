from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.stat_test as stat_test
def run(ds:Dataset.DataSplits, model_config:Config.OldModel, train_flag:bool, base_type, clip_processor):
    
    if base_type == 'svm':
        base_model = Model.svm(ds.loader['train_clip'], clip_processor)
    else:
        base_model = Model.resnet(2)
    if train_flag:
        if base_type == 'svm':
            base_model.train(ds.loader['train_clip'])
        else:
            base_model.train(ds.loader['train'], ds.loader['val'])
    else:
        base_model.load(model_config)
    # Evaluate
    gt,pred,_  = base_model.eval(ds.loader['test'])
    acc = (gt==pred).mean()*100
    gt,pred,_  = base_model.eval(ds.loader['test_shift'])
    acc_shift = (gt==pred).mean()*100

    clf = Detector.SVM(ds.loader['train_clip'], clip_processor)
    score = clf.fit(base_model, ds.loader['val_shift'])
    _, detect_prec = clf.predict(ds.loader['val_shift'], compute_metrics=True, base_model=base_model)
    if train_flag:
        base_model.save(model_config.path)

    shift_score = Subset.mis_cls_stat('val_shift', ds, base_model)

    stat_test.run(clf, ds.loader['val_shift'], base_model, model_config)

    return acc, acc_shift, detect_prec, shift_score

def main(epochs,  model_dir ='', train_flag=False, device_id=0, base_type=''):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    acc_list, acc_shift_list, detect_prec_list, shift_list = [], [], [], []
    clip_processor = Detector.load_clip(device_config)
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, base_type)
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        prec, acc_shift, detect_prec, shift_score = run(ds, old_model_config, train_flag, base_type, clip_processor)
        acc_list.append(prec)
        acc_shift_list.append(acc_shift)
        detect_prec_list.append(detect_prec)
        shift_list.append(shift_score)
    print('Model Average Acc before shift: {}%'.format(np.round(np.mean(acc_list),decimals=3)))
    print('Model Average Acc after shift: {}%'.format(np.round(np.mean(acc_shift_list),decimals=3)))
    Detector.statistics(detect_prec_list, 'precision')
    print('Shifted Data Proportion on Model Misclassifications: {}%'.format(np.round(np.mean(np.array(shift_list),axis=0), decimals=3)))

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
    parser.add_argument('-bt','--base_type',type=str,default='resnet')

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir,args.train_flag, args.device, args.base_type)