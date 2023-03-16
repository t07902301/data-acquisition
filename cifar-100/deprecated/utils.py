import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torchvision
import sys
sys.path.append('..')
import failure_directions
import numpy as np
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
# from tqdm import tqdm
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
# import copy
# import pickle
import yaml
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    file.close()
import sklearn.metrics as sklearn_metrics
# from imblearn.over_sampling import SMOTE,ADASYN
from copy import deepcopy

device = config["device"]
torch.cuda.set_device(device)
ds_root = config['ds_root']

hparams = config['hparams']
mean = config['mean']
std = config['std']
num_workers = config['num_workers']

build_fn = failure_directions.model_utils.BUILD_FUNCTIONS[hparams['arch_type']]

# model_path_root = os.path.join(config["model_path_root"],str(hparams['batch_size']))
# if os.path.exists(model_path_root) is False:
#     os.makedirs(model_path_root)
    
remove_fine_labels = [4, 73, 54, 10, 51, 40, 84, 18, 3, 12, 33, 38, 64, 45, 2, 44, 80, 96, 13, 81]

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=np.array(mean)/255, std=np.array(std)/255)])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    base_transform
])


# For visualization
INV_NORM = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [255/x for x in std]),
                                transforms.Normalize(mean = [-x /255 for x in mean],
                                                     std = [ 1., 1., 1. ])])
TOIMAGE = transforms.Compose([INV_NORM, transforms.ToPILImage()])

generator = torch.Generator()
generator.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(0)
import cifar

def dummy_acquire(cls_gt, cls_pred,method,img_num):

    if method == 'hard':
        result_mask = cls_gt != cls_pred
    else:
        result_mask = cls_gt == cls_pred
    result_mask_indices = np.arange(len(result_mask))[result_mask] # for this class
    if result_mask.sum() > img_num:
        new_img_indices = np.random.choice(result_mask_indices,img_num,replace=False) # points to the origin dataset
    else:
        print('class result_mask_indices with a length',len(result_mask_indices))
        new_img_indices = result_mask_indices
    return new_img_indices    

def get_weights(labels,cls_num=2):
    # cls_num = gts.max() + 1
    weights = []
    c_num_total = len(labels)
    labels = np.array(labels)
    for c in range(cls_num):
        c_num = np.sum(labels==c)
        print('class {}: {}'.format(c,c_num))
        # if c_num == 0:
        #     return torch.ones([cls_num]).cuda()
        # weights.append(1/c_num)
        weights.append(1-c_num/c_num_total)
        # assert (weights >= 0).all().item()
    return torch.tensor(weights).float().cuda() 

def get_ds_labels(ds):
    ds = list(ds)
    labels = [pair[1] for pair in ds]
    return labels

def save_model(model, path):
    torch.save({
        'build_params': model._build_params,
        'state_dict': model.state_dict(),
    }, path)
    print('model saved to {}'.format(path))

def top_dv(sorted_indices, K=0, clf='SVM'):
    '''
    return indices of images with top decision scores
    '''
    # img_list = [ds[sorted_indices[i]] for i in range(K)]
    # return img_list
    if clf == 'SVM':
        dv_indices = sorted_indices[:K]
    else:
        dv_indices = sorted_indices[::-1][:K] # decision scores from model confidence is non-negative
    return list(dv_indices)

def tune(base_model,train_loader,val_loader,weights=None,log_model=False,model_save_path=''):
    training_args=hparams['tuning']
    training_args['iters_per_epoch'] = len(train_loader)
    training_args['optimizer'] = hparams['optimizer']
    trainer = failure_directions.LightWeightTrainer(training_args=training_args,
                                                    exp_name=model_save_path, enable_logging=log_model,
                                                    bce=False, set_device=True,weights=weights)
    best_model = trainer.fit(base_model, train_loader, val_loader)
    return best_model # check point during refinig
    # return base_model
    
def build_model(num_class=hparams['num_classes'],use_pretrained=False):
    model = build_fn(hparams['arch'], num_class,use_pretrained)
    model = model.cuda()
    return model

def train_model(train_loader,val_loader,num_class=hparams['num_classes'],use_pretrained=False,log_model=False,model_save_path=''):
    '''return the last checkpoint'''
    model = build_model(num_class,use_pretrained)
    training_args=hparams['training']
    training_args['optimizer'] = hparams['optimizer']
    training_args['iters_per_epoch'] = len(train_loader)
    trainer = failure_directions.LightWeightTrainer(training_args=hparams['training'],
                                                    exp_name=model_save_path, enable_logging=log_model,
                                                    bce=False, set_device=True)
    _ = trainer.fit(model, train_loader, val_loader)
    return model # the last ckpnt  

def load_model(path,use_pretrained=False):
    out = torch.load(path,map_location=device)
    model = build_fn(**out['build_params'],use_pretrained=use_pretrained) #test model or fine-tune model not feature extraction
    model.load_state_dict(out['state_dict'])
    model = model.cuda()
    print('load model from {}'.format(path))
    # print(out['run_metadata'])
    return model
def split_dataset(targets, num_classes, selected_classes, split_ratio=0 ):
    '''split the dataset into two parts. in Part I: take every split_amt data-point per class.
    in part II: put the rest of the points
    
    Params: 
           targets: numpy array of target labels
           num_classes: number of classes
           split_ratio: how much to split into PART 1 from the dataset 
    Return:
           p1_indices: indices belonging to part 1 (split_ratio)
           p2_indices: indices belonging to part 2
    '''
    
    if torch.is_tensor(targets):
        targets = targets.numpy()
    N = len(targets)
    p1_indices = []
    if len(selected_classes) == 0:
        selected_classes = [i for i in range(num_classes)]
    for c in selected_classes:
        cls_indices = np.arange(N)[targets == c]
        p1_indices.append(np.random.choice(cls_indices,int(split_ratio*len(cls_indices)),replace=False))
    p1_indices = np.concatenate(p1_indices)
    mask_p1 = np.zeros(N, dtype=bool)
    mask_p1[p1_indices] = True
    mask_p2 = ~mask_p1 
    p2_indices = np.arange(N)[mask_p2]
    assert len(p1_indices) + len(p2_indices) == N
    # return torch.tensor(p1_indices), torch.tensor(p2_indices)
    return p1_indices,p2_indices

def remove_data(train_ds,percent):
    removed_dict = config['removed_subclass']
    subclass_indices = {}
    new_train_ds = []
    for idx in range(20):
        subclass_indices[idx] = []
    for idx,info in enumerate(train_ds):
        coarse_label = info[2]
        fine_label = info[1]
        if fine_label == removed_dict[str(coarse_label)]:
            subclass_indices[coarse_label].append(idx)
    for coarse_label,indices in subclass_indices.items():
        indices_num = len(indices)
        removed_num = int(indices_num*percent)
        subclass_subset = torch.utils.data.Subset(train_ds,indices)
        subclass_left,_ = torch.utils.data.random_split(subclass_subset,[indices_num-removed_num,removed_num],generator=generator) 
        new_train_ds += list(subclass_left)
    return new_train_ds  
def get_minority_subset(test_ds):
    removed_dict = config['removed_subclass']
    subset_indices = []
    for idx,info in enumerate(test_ds):
        coarse_label = info[2]
        fine_label = info[1]
        if fine_label == removed_dict[str(coarse_label)]:
            subset_indices.append(idx)  
    return torch.utils.data.Subset(test_ds,subset_indices)

ce = nn.CrossEntropyLoss(reduction='none')
def evaluate_model(dataloader,model):
    with torch.no_grad():
        with autocast():
            gts, preds, confs = [], [], []
            for x, y,fine_y in dataloader:
                x = x.cuda()
                logits = model(x)
                gts.append(y.cpu())
                preds.append(logits.argmax(-1).cpu())
                softmax_logits = nn.Softmax(dim=-1)(logits) # logits: unnormalized output before the last layer
                # loss.append(ce(softmax_logits,y.cuda()))
                confs.append(softmax_logits[torch.arange(logits.shape[0]), y].cpu())

    gts = torch.cat(gts).numpy()
    preds = torch.cat(preds).numpy()
    confs = torch.cat(confs).numpy()
    return gts, preds,confs
    # return {
    #     'gt': gts, 
    #     'preds': preds, 
    #     'confs': confs
    # }
def minority_in_ds(ds):
    '''
    return #minority in ds
    '''
    cnt = 0
    for info in ds:
        if info[2] in remove_fine_labels:
            cnt += 1
    return cnt
def create_dataset_split(train_ds,aug_train_ds,test_ds,select_fine_labels=None):
    # When all classes are used, only work on removal
    # When some classes are neglected, test set and the big train set will be shrank.
    train_size = config["train_size"]
    val_size = config["val_size"]
    market_size = config["market_size"]
    remove_rate = config['remove_rate']

    train_fine_labels = np.array([info[2] for info in train_ds])
    train_fine_labels_cls_num = train_fine_labels.max()+1

    if select_fine_labels == None:
        select_fine_labels = []

    if len(select_fine_labels)>0:
        train_select_indice = []
        for c in select_fine_labels:
            train_select_indice.append(np.arange(len(train_ds))[c==train_fine_labels])
        train_select_indice = np.concatenate(train_select_indice)

        test_fine_labels = np.array([info[2] for info in test_ds])
        test_subset_indices = []
        for c in select_fine_labels:
            test_subset_indices.append(np.arange(len(test_ds))[test_fine_labels == c]) 
        test_subset_indices = np.concatenate(test_subset_indices)   

        train_ds = torch.utils.data.Subset(train_ds,train_select_indice) 
        aug_train_ds = torch.utils.data.Subset(aug_train_ds,train_select_indice) 
        test_ds = torch.utils.data.Subset(test_ds,test_subset_indices) 
        train_fine_labels = np.array([info[2] for info in train_ds])
        test_fine_labels = np.array([info[2] for info in test_ds])


    train_indices,val_market_indices = split_dataset(train_fine_labels,train_fine_labels_cls_num,split_ratio=train_size/(train_size+val_size+market_size),selected_classes=select_fine_labels)

    val_market_set = torch.utils.data.Subset(train_ds,val_market_indices)
    aug_val_market_set = torch.utils.data.Subset(aug_train_ds,val_market_indices)
    clip_train_ds_split = torch.utils.data.Subset(train_ds,train_indices)
    aug_train_ds_split = torch.utils.data.Subset(aug_train_ds,train_indices)

    val_mar_fine_labels = np.array([info[2]  for info in val_market_set])
    
    val_indices,market_indices = split_dataset(val_mar_fine_labels,train_fine_labels_cls_num,split_ratio=val_size/(val_size+market_size),selected_classes=select_fine_labels)

    val_ds = torch.utils.data.Subset(val_market_set,val_indices)
    market_ds = torch.utils.data.Subset(val_market_set,market_indices)
    aug_market_ds = torch.utils.data.Subset(aug_val_market_set,market_indices)

    train_fine_labels = np.array([info[2]  for info in aug_train_ds_split])
    _, left_indices = split_dataset(train_fine_labels,train_fine_labels_cls_num,split_ratio=remove_rate,selected_classes=remove_fine_labels)
    left_clip_train = torch.utils.data.Subset(clip_train_ds_split,left_indices)
    left_aug_train = torch.utils.data.Subset(aug_train_ds_split,left_indices)

    ds = {}
    ds['train'] =  left_aug_train
    ds['val'] =  val_ds
    ds['market'] =  market_ds
    ds['train_clip'] = left_clip_train
    ds['market_aug'] =  aug_market_ds
    ds['test'] = test_ds

    return ds


def ds_to_list(ds_1,ds_2):
    '''
    change Dataset type to list
    '''

    img_list_1 = []
    for img_pair in ds_1:
        img_list_1.append(img_pair[0])
    img_list_2 = []
    for img_pair in ds_2:
        img_list_2.append(img_pair[0])

    return img_list_1,img_list_2

def modify_labels(ds,c0):
    '''
    c0: the class label to be modified to class 0
    '''
    dataset = []
    for i,img in enumerate(ds):
        label = 0 if img[1]== c0 else 1 
        dataset.append((img[0],label))
    return dataset

def get_metrics(gt, pred, metric, check_class=None, confusion_matrix=False):
    acc = []
    cls_list = [check_class] if check_class!= None else [i for i in range(gt.max()+1)]
    print(metric)
    for c in cls_list: # go through metric for every class
        if metric == 'precision':
            mask = pred == c
        else:
            mask = gt == c
        masked_gt = gt[mask]
        masked_pred = pred[mask]
        masked_len = masked_pred.size(dim=0)
        metric = ((masked_gt==masked_pred).sum(dim=0).item())/masked_len*100
        acc.append(metric)  
    if confusion_matrix:
        print(sklearn_metrics.confusion_matrix(gt,pred))
    return acc 
def get_CLF(base_model,dataloaders,svm_fit_label= 'val',metric=''):
    clip_processor = failure_directions.CLIPProcessor(ds_mean=mean, ds_std=std)
    svm_fit_gt,svm_fit_pred,_ = evaluate_model(dataloaders[svm_fit_label],base_model) # gts, preds, loss
    clip_features = {}
    for split, loader in dataloaders.items():
        if (split == 'train') or (split=='market_aug'):
            continue
        clip_features[split] = clip_processor.evaluate_clip_images(loader)
    svm_fitter = failure_directions.SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv= config['clf_args']['k-fold'])
    svm_fitter.set_preprocess(clip_features['train_clip'])
    score = svm_fitter.fit(preds=svm_fit_pred, gts=svm_fit_gt, latents=clip_features[svm_fit_label])

    return svm_fitter,clip_features,score

