from utils.strategy import *
from utils.set_up import set_up, CLF_statistics
def run(ds:DataSplits, model_config:OldModelConfig, train_flag:bool):
    if train_flag:
        base_model = train_model(ds.loader['train'], ds.loader['val'], num_class=model_config.class_number)
    else:
        base_model = load_model(model_config)

    # Evaluate
    gt,pred,_  = evaluate_model(ds.loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    # Get SVM 
    clf,clip_features,clf_score = get_CLF(base_model,ds.loader)
    if train_flag:
        # Get a new base model from Resnet
        save_model(base_model, model_config.path)
    return base_acc,clf_score

def main(epochs,  model_dir ='', train_flag=False, device=0):
    # batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, _ = parse_config(model_dir, False)
    # print_config(batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio)
    # ds_list = get_data_splits_list(epochs, select_fine_labels, label_map, ratio)
    # device_config = 'cuda:{}'.format(device)
    # torch.cuda.set_device(device_config)
    batch_size, select_fine_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, model_dir, False, device)

    acc_list, clf_score_list = [], []
    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds.get_dataloader(batch_size)
        old_model_config = OldModelConfig(batch_size,superclass_num,model_dir, device_config, epo)
        acc, clf_score = run(ds, old_model_config, train_flag)
        acc_list.append(acc)
        clf_score_list.append(clf_score)
    print(np.mean(acc_list))
    CLF_statistics(epochs, clf_score_list)
    split_data = ds.dataset
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.model_dir,args.train_flag, args.device)
    # print(args.method)