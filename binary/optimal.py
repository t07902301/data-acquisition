from utils.strategy import *
from utils.set_up import set_up
def run(ds:Dataset.DataSplits, model_config:Config.OldModel):
    base_model = Model.train(ds.loader['train'], ds.loader['val'], num_class=2)
    gt,pred,_  = Model.evaluate(ds.loader['test'],base_model)
    base_acc_shift = (gt==pred).mean()*100
    # shift_score = Model.shift_importance(ds.dataset['test'], gt, pred)
    shift_score = 0
    return base_acc_shift, shift_score

def main(epochs,  model_dir ='', device=0):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    acc_list, clf_score_list, clf_prec_list, shift_score_list, acc_shift_list = [], [], [], [], []
    recall_list = []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        ds = ds_list[epo]
        ds = Dataset.DataSplits(ds, old_model_config.batch_size)
        acc_shift, shift_score = run(ds, old_model_config)
        acc_shift_list.append(acc_shift)
        shift_score_list.append(shift_score)
    print('Model Average Acc:', np.round(np.mean(acc_shift_list),decimals=3))
    print(acc_shift_list)
    # CLF.statistics(clf_score_list, 'score')
    # print(clf_score_list)
    # CLF.statistics(clf_prec_list, 'precision')
    # print(clf_prec_list)
    print('Distribution Shift Proportion on Model Misclassifications', np.round(np.mean(np.array(shift_score_list),axis=0), decimals=3))
    
    # print(check(ds.dataset['test_shift']))
    # print(check(ds.dataset['val_shift']))
    # print(check(ds.dataset['val']))
    # print(check(ds.dataset['train']))
    # print(check(ds.dataset['test']))
    
    split_data = ds.dataset
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir, args.device)