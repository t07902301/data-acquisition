from src.utils import *


def run(new_img_num,epoch,method,test_loader,base_model,new_model_setter='refine'):

    gt,pred,_  = evaluate_model(test_loader,base_model)
    base_acc = (gt==pred).mean()*100

    new_model_path_root = os.path.join(model_path_root,new_model_setter)
    path = os.path.join(new_model_path_root,'{}_{}_{}.pt'.format(method,new_img_num,epoch))
    new_model = load_model(path)
    new_model = new_model.eval()
    gt,pred,_  = evaluate_model(test_loader,new_model)
    new_acc = (gt==pred).mean()*100
    return new_acc-base_acc
    
def main(epochs,train_flag,metric=''):

    acc_change_list = []
    
    train_ds = cifar.CIFAR100(ds_root, train=True,transform=base_transform,coarse=True)
    test_ds = cifar.CIFAR100(ds_root, train=False,transform=base_transform,coarse=True)
    aug_train_ds = cifar.CIFAR100(ds_root, train=True,transform=train_transform,coarse=True)

    img_per_cls_list = [25,50,75,100]
    # img_per_cls_list = [25]
    method_list = ['dv','sm','conf','mix','seq','seq_clf']
    # method_list = ['seq','seq_clf']

    # select_fine_labels = [30, 1, 62, 9, 0, 22, 5, 6, 42, 17, 23, 15, 34, 26, 11, 27, 36, 47, 8, 41, 4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]
    select_fine_labels = None
    # cls_num = 20 if select_fine_labels==None else len(select_fine_labels)//2

    for i in range(epochs):
        print('epoch',i)
        ds = create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=select_fine_labels)

        subset_loader = {
            k: torch.utils.data.DataLoader(ds[k], batch_size=hparams['batch_size'], shuffle=(k=='train'), drop_last=(k=='train'),num_workers=num_workers)
            for k in ds.keys()
        }
        path = os.path.join(model_path_root,'{}.pt'.format(i))
        base_model = load_model(path)
        base_model = base_model.eval()

        acc_change_epoch = []
        for method in method_list:
            acc_change_method = []
            for img_cls in img_per_cls_list:
                acc_change = run(img_cls,i,method,test_loader=subset_loader['test'],base_model=base_model,new_model_setter='retrain')
                acc_change_method.append(acc_change)
            acc_change_epoch.append(acc_change_method)
        acc_change_list.append(acc_change_epoch)

    acc_change_list = np.round(acc_change_list,decimals=3)
    # print(*acc_change_list.tolist(),sep='\n')
    # print('acc change average')
    # for m_idx,method in enumerate(method_list): 
    #     method_result = acc_change_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
    #     method_avg = np.round(np.mean(method_result,axis=0),decimals=3)
    #     print('method:', method)
    #     print(*method_avg,sep=',')
    method_labels = ['greedy decision value','random sampling','model confidence','greedy+sampling','sequential','sequential with only SVM updates']
    for m_idx,method in enumerate(method_labels): 
        method_result = acc_change_list[:,m_idx,:] #(e,m,img_num) e:epochs,m:methods
        method_avg = np.round(np.mean(method_result,axis=0),decimals=3) #(e,img_num)
        plt.plot(img_per_cls_list,method_avg,label=method)
    plt.xticks(img_per_cls_list,img_per_cls_list)
    plt.xlabel('#new images for each superclass')
    plt.ylabel('#model accuracy change')
    plt.legend()
    plt.savefig('figure/base.png')
    plt.clf()


        

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=10)
    parser.add_argument('-tf','--train_flag',type=bool,default=False)
    parser.add_argument('-m', '--metric',type=str,default='precision')

    args = parser.parse_args()
    # method, img_per_cls, save_model
    main(args.epochs,args.train_flag,metric=args.metric)
    # print(args.method)