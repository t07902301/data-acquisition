from utils.strategy import *
from utils import config
def run(ds, model_config, train_flag):
    loader = get_dataloader(ds, model_config.batch_size)

    if train_flag:
        base_model = train_model(loader['train'],loader['val'],num_class=model_config.class_number)
    else:
        base_model = load_model(model_config.path)
    # Evaluate
    gt,pred,_  = evaluate_model(loader['test'],base_model)
    base_acc = (gt==pred).mean()*100

    # Get SVM 
    clf,clip_features,clf_score = get_CLF(base_model,loader)
    if train_flag:
        # Get a new base model from Resnet
        save_model(base_model, model_config.path)
    return base_acc,clf_score

def main(epochs, train_flag, model_dir):
    hparams = config['hparams']
    batch_size = hparams['batch_size'][model_dir]

    data_config = config['data']
    select_fine_labels = data_config['selected_labels'][model_dir]
    label_map = data_config['label_map'][model_dir]

    percent_list = []
    superclass_num = int(model_dir.split('-')[0])
    base_acc_list = []
    cv_scores_list = []
    val_score_list = []
    init_acc_list = []

    for i in range(epochs):
        print('epoch',i)
        ds = create_dataset_split(data_config['ds_root'],select_fine_labels,model_dir)
        if select_fine_labels!=[]:
            modify_coarse_label(ds,label_map)

        model_config = OldModelConfig(batch_size,superclass_num,model_dir, i)
                
        base_acc,score = run(ds, model_config, train_flag)
        base_acc_list.append(base_acc)
        val_score_list.append(score['fit'] )
        cv_scores_list.append(score['cv'])

    for split,split_ds in ds.items():
        print(split,len(split_ds))

    print('base_acc')
    # print(np.round(np.mean(base_acc_rslts,axis=0),decimals=3))
    print(np.round(base_acc_list,decimals=3),sep=',')

    print('base acc avg')
    print(np.round(np.mean(base_acc_list,axis=0),decimals=3))

    # print('init acc avg')
    # print(np.round(np.mean(init_acc_list,axis=0),decimals=3))

    # print('val scores')
    # print(np.round(val_score_list,decimals=3))
    # print('val scores avg')
    # print(np.round(np.mean(val_score_list,axis=0),decimals=3))

    print('cv scores avg')
    print(np.round(np.mean(cv_scores_list,axis=0),decimals=3))
    print(data_config)
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-d','--model_dir',type=str,default='')

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.train_flag,args.model_dir)
    # print(args.method)