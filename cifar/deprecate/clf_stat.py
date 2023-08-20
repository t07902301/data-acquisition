from utils.strategy import *
from utils.set_up import set_up

def get_new_clf_statistics(new_img_num_list, new_model_config:Config.NewModel, acquire_instruction:Config.Acquistion, data_splits:Dataset.DataSplits, base_model):
    '''
    base model and its clf stat from different number of new data. 
    '''
    clf_precision_list = []
    if new_model_config.pure:
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items('seq_clf',new_img_num)
            clf = Log.get_log_clf(acquire_instruction, new_model_config, data_splits.loader['train_clip'])
            _, precision = clf.predict(data_splits.loader['test_shift'], compute_metrics=True, base_model = base_model)
            clf_precision_list.append(precision)
    return clf_precision_list

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0):
    print('Use pure: ',pure)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    precision_list, base_precision_list = [], []

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment=False)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)        

        base_model = Model.load(old_model_config)
        clf = CLF.SVM(dataset_splits.loader['train_clip'])
        _ = clf.fit(base_model, dataset_splits.loader['val_shift'])
        _, base_precision = clf.predict(dataset_splits.loader['test_shift'], compute_metrics=True, base_model=base_model)
        base_precision_list.append(base_precision)

        prec_epo = get_new_clf_statistics(new_img_num_list, new_model_config, acquire_instruction, dataset_splits, base_model)
        precision_list.append(prec_epo)

    print("Before acquisition:")
    CLF.statistics(base_precision_list, 'precision')
    print("After acquisition:")
    CLF.statistics(precision_list, 'precision', new_img_num_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str)
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure, model_dir=args.model_dir, device=args.device)