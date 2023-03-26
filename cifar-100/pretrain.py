from utils.strategy import *
from utils import config
def run(ds:DataSplits, model_config:OldModelConfig, train_flag:bool):
    if train_flag:
        base_model = train_model(ds.loader['train'], ds.loader['val'], num_class=model_config.class_number)
    else:
        base_model = load_model(model_config.path)

    # Evaluate
    gt,pred,_  = evaluate_model(ds.loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    # Get SVM 
    clf,clip_features,clf_score = get_CLF(base_model,ds.loader)
    if train_flag:
        # Get a new base model from Resnet
        save_model(base_model, model_config.path)
    return base_acc,clf_score

def main(epochs,  model_dir ='', train_flag=False):
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num = parse_config(model_dir, False)

    acc_list, clf_score_list = [], []

    for epo in range(epochs):

        print('in epoch {}'.format(epo))

        ds = DataSplits(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            ds.modify_coarse_label(label_map)
        ds.get_dataloader(batch_size)
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, epo)
        acc, clf_score = run(ds, old_model_config, train_flag)
        acc_list.append(acc)
        clf_score_list.append(clf_score['cv'])

    print(np.mean(acc_list), np.mean(clf_score_list))
    print(data_config)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.model_dir,args.train_flag)
    # print(args.method)