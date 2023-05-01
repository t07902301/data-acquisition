from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter

def get_new_clf_statistics(new_img_num_list, new_model_config:Config.NewModel, acquire_instruction:Config.Acquistion):
    '''
    base model and its clf stat from different number of new data. 
    '''
    cv_score_list = []
    clf_precision_list = []
    if new_model_config.pure:
        for new_img_num in new_img_num_list:
            acquire_instruction.set_items('seq_clf',new_img_num)
            stat_config = Log.get_config(new_model_config, acquire_instruction, 'stat')
            clf_stat = Log.load(stat_config)
            cv_score_list.append(clf_stat['cv score'])
            clf_precision_list.append(clf_stat['precision'])
    return cv_score_list, clf_precision_list

def shift_intervention():
    pass

def main(epochs, new_model_setter='retrain', pure=False, model_dir ='', device=0):
    print('Use pure: ',pure)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, pure, device)
    score, precision = [], []
    base_score, base_precision = [], []

    ds_list = Dataset.get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, pure, new_model_setter,augment=False)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        
        base_model = Model.load(old_model_config)
        # Evaluate
        gt,pred,_  = Model.evaluate(ds.loader['test'],base_model)
        clf,clip_features,base_score_epo = CLF.get_CLF(base_model, ds.loader)
        _, base_prec_epo = CLF.apply_CLF(clf, ds.loader['test'], clip_features, preds=pred, compute_metrics=True)
        base_score.append(base_score_epo)
        base_precision.append(base_prec_epo)

        score_epo, prec_epo = get_new_clf_statistics(new_img_num_list, new_model_config, acquire_instruction)
        score.append(score_epo)
        precision.append(prec_epo)

    if pure:
        print("Before acquisition:")
        CLF.statistics(base_score, base_precision)
        print("After acquisition:")
        CLF.statistics(score, precision, new_img_num_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=True)
    parser.add_argument('-md','--model_dir',type=str)
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs,pure=args.pure, model_dir=args.model_dir, device=args.device)