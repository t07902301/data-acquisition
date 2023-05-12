from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
def run(ds:Dataset.DataSplits, model_config:Config.OldModel, train_flag:bool):
    if train_flag:
        base_model = Model.train(ds.loader['train'], ds.loader['val'], num_class=2)
    else:
        base_model = Model.load(model_config)
    # Evaluate
    gt,pred,_  = Model.evaluate(ds.loader['test'],base_model)
    base_acc = (gt==pred).mean()*100
    gt,pred,_  = Model.evaluate(ds.loader['test_shift'],base_model)
    base_acc_shift = (gt==pred).mean()*100
    cls_mask = (gt==0)
    recall = ((gt==pred)[cls_mask]).mean()*100
    clf = CLF.SVM(ds.loader['train_clip'])
    score = clf.fit(base_model, ds.loader['val_shift'])
    _, precision = clf.predict(ds.loader['val_shift'], compute_metrics=True, base_model=base_model)
    if train_flag:
        # Get a new base model from Resnet
        Model.save(base_model, model_config.path)
    shift_score = Subset.shift_importance('val_shift', ds, base_model)
    return base_acc, score, precision, shift_score, base_acc_shift, recall 

def main(epochs,  model_dir ='', train_flag=False, device=0):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    acc_list, clf_score_list, clf_prec_list, shift_score_list, acc_shift_list = [], [], [], [], []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        acc, score, prec, shift_score, acc_shift, recall = run(ds, old_model_config, train_flag)
        acc_list.append(acc)
        clf_score_list.append(score)
        clf_prec_list.append(prec)
        shift_score_list.append(shift_score)
        acc_shift_list.append(acc_shift)
    # print( np.round(np.mean(recall_list),decimals=3))
    print('Model Average Acc before shift: {}%'.format(np.round(np.mean(acc_list),decimals=3)))
    print(acc_list)
    print('Model Average Acc after shift: {}%'.format(np.round(np.mean(acc_shift_list),decimals=3)))
    print(acc_shift_list)
    print('In shifted val set:')
    CLF.statistics(clf_score_list, 'score')
    print(clf_score_list)
    CLF.statistics(clf_prec_list, 'precision')
    print(clf_prec_list)
    print('Distribution Shift Proportion on Model Misclassifications: {}%'.format(np.round(np.mean(np.array(shift_score_list),axis=0), decimals=3)))
    # print(shift_score_list)
    
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