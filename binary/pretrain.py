from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.data as DataStat
def run(ds:Dataset.DataSplits, model_config:Config.OldModel, train_flag:bool, detect_instruction:Config.Detection):
    base_model = Model.prototype_factory(model_config.base_type, model_config.class_number, detect_instruction.vit)
    if train_flag:
        if model_config.base_type == 'svm':
            base_model.train(ds.loader['train'])
        else:
            base_model.train(ds.loader['train'], ds.loader['val'])
    else:
        base_model.load(model_config.path, model_config.device)
    # Evaluate
    acc = base_model.acc(ds.loader['test'])
    acc_shift = base_model.acc(ds.loader['test_shift'])
    clf = Detector.factory(detect_instruction.name, clip_processor = detect_instruction.vit, split_and_search=True, data_transform='clip')
    _ = clf.fit(base_model, ds.loader['val_shift'], ds.dataset['val_shift'], model_config.batch_size)
    _, detect_prec = clf.predict(ds.loader['val_shift'], compute_metrics=True, base_model=base_model)
    print('In fitting CLF:', detect_prec)
    gt,pred,_  = base_model.eval(ds.loader['val_shift'])
    print('Val Shift Acc:', (gt==pred).mean()*100)
    _, detect_prec = clf.predict(ds.loader['test_shift'], compute_metrics=True, base_model=base_model)
    print('In testing CLF:', detect_prec)
    print('Test Shift Acc:', acc_shift)

    if train_flag:
        base_model.save(model_config.path)

    shift_score = DataStat.mis_label_stat('val_shift', ds, base_model)

    return acc, acc_shift, detect_prec, shift_score

def main(epochs,  model_dir ='', train_flag=False, device_id=0, base_type='', detector_name=''):
    print('Detector Name:', detector_name)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, device_id)
    acc_list, acc_shift_list, detect_prec_list, shift_list = [], [], [], []
    clip_processor = Detector.load_clip(device_config)
    detect_instrution = Config.Detection(detector_name, clip_processor)
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, base_type)
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        prec, acc_shift, detect_prec, shift_score = run(ds, old_model_config, train_flag, detect_instrution)
        acc_list.append(prec)
        acc_shift_list.append(acc_shift)
        detect_prec_list.append(detect_prec)
        shift_list.append(shift_score)
    print('Old Model Acc before shift: {}%'.format(np.round(np.mean(acc_list),decimals=3)))
    print('Old Model Acc after shift: {}%'.format(np.round(np.mean(acc_shift_list),decimals=3)))
    print('Shifted Data Proportion on Old Model Misclassifications: {}%'.format(np.round(np.mean(np.array(shift_list),axis=0), decimals=3)))
    Detector.statistics(detect_prec_list, 'precision')

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
    parser.add_argument('-bt','--base_type',type=str,default='resnet_1')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')

    args = parser.parse_args()
    main(args.epochs,args.model_dir,args.train_flag, args.device, args.base_type, args.detector_name)