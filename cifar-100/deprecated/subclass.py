from utils import *
def run(ds = None,epoch=0,train_flag=False,cls_num=None):
    subset_loader = {
        k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], shuffle=(k=='train'), drop_last=(k=='train'),num_workers=num_workers)
        for k in ds.keys()
    }

    path = os.path.join(model_path_root,'{}.pt'.format(epoch))

    # init_model = get_init_model()
    # init_model = init_model.eval()
    # gt,pred,_  = evaluate_model(subset_loader['test'],init_model)
    # init_acc = (gt==pred).mean()*100

    if train_flag:
        base_model = train_model(subset_loader['train'],subset_loader['val'],num_class=cls_num,use_pretrained=False)
        # base_model = tune(subset_loader['train'],subset_loader['val'])
    else:
        base_model = load_model(path)
    # Evaluate
    base_model = base_model.eval()
    gt,pred,_  = evaluate_model(subset_loader['test'],base_model)
    # base_acc = get_metrics(gt,pred,metric, confusion_matrix=True)
    # base_acc = (gt==pred)[test_sub_indices].mean()*100
    base_acc = (gt==pred).mean()*100
    print(base_acc)
    for c in range(cls_num):
        mask = gt==c
        print((gt[mask]==pred[mask]).mean()*100)
    # Get SVM 
    clf,clip_features,clf_score = get_CLF(base_model,subset_loader)
    if train_flag:
        # Get a new base model from Resnet
        save_model(base_model, path)
    return base_acc,clf_score
def visualize_images(ds, ds_indices, dv, K=10, path=''):
    fig, ax = plt.subplots(1, K)
    for i in range(K):
        idx = ds_indices[i]
        ax[i].imshow(TOIMAGE(ds[idx][0]))
        ax[i].axis(False)
        # ax[i].set_title(f"{dv[idx]:0.3f}")
    # plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)
def main(epochs,train_flag):
    label = 0
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    vis_list = []
    for idx,info in enumerate(train_ds):
        if info[1] == label:
            vis_list.append(idx)
            if len(vis_list) > 5:
                break
    visualize_images(train_ds,vis_list,None,5,'figure/train_demo.png')
    vis_list = []
    for idx,info in enumerate(test_ds):
        if info[1] == label:
            vis_list.append(idx)
            if len(vis_list) > 5:
                break
    visualize_images(test_ds,vis_list,None,5,'figure/test_demo.png')
# def main(epochs,train_flag):
#     base_acc_list = []
#     cv_scores_list = []
#     val_score_list = []
#     init_acc_list = []
    
#     train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
#     test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
#     aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)

#     ## Subset of test set
#     # test_fine_labels = np.array([info[2] for info in test_ds])
#     # test_ds_len = len(test_ds)
#     # test_subset_indices = []
#     # for c in remove_fine_labels:
#     #     test_subset_indices.append(np.arange(test_ds_len)[test_fine_labels == c])
#     # test_subset_indices = np.concatenate(test_subset_indices)
#     # # test_ds = torch.utils.data.Subset(test_ds,test_subset_indices)    
#     # ds ={
#     #     'test':test_ds
#     # }

#     select_fine_labels = [30, 4, 9, 10, 0, 51]
#     sp_labels_map = {
#         0: 0,
#         3: 1,
#         4: 2
#     }

#     for i in range(epochs):

#         print('epoch',i)

#         ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)
#         for split,infos in ds.items():
#             new_ds_split = []
#             for info in infos:
#                 new_ds_split.append((info[0],sp_labels_map[info[1]],info[2]))
#             ds[split] = new_ds_split


#         base_acc,score = run(ds=ds,epoch=i,train_flag=train_flag,cls_num=3)
#         base_acc_list.append(base_acc)
#         val_score_list.append(score['fit'] )
#         cv_scores_list.append(score['cv'])

#     for split,split_ds in ds.items():
#         print(split,len(split_ds))

#     print('base_acc')
#     # print(np.round(np.mean(base_acc_rslts,axis=0),decimals=3))
#     print(np.round(base_acc_list,decimals=3),sep=',')

#     print('base acc avg')
#     print(np.round(np.mean(base_acc_list,axis=0),decimals=3))

#     # print('init acc avg')
#     # print(np.round(np.mean(init_acc_list,axis=0),decimals=3))

#     # print('val scores')
#     # print(np.round(val_score_list,decimals=3))
#     print('val scores avg')
#     print(np.round(np.mean(val_score_list,axis=0),decimals=3))

#     print('cv scores avg')
#     print(np.round(np.mean(cv_scores_list,axis=0),decimals=3))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.train_flag)
    # print(args.method)