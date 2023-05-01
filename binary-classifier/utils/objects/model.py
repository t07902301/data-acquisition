import sys
sys.path.append('..')
import failure_directions.src.model_utils as model_utils
import failure_directions.src.trainer as trainer_utils
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from utils.env import model_env
from utils import config
from utils.objects.Config import OldModel
from utils.acquistion import extract_class_indices

model_env()
hparams = config['hparams']
import numpy as np       
import torch.nn.functional as F
def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)
# Neural Network
class Network(nn.Module):
    # Network Initialisation
    def __init__(self, params):
        
        super(Network, self).__init__()
    
        Cin,Hin,Win=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        num_classes=params["num_classes"] 
        self.dropout_rate=params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        # self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        # h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten= h*w*4*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        # X = F.relu(self.conv4(X))
        # X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X=F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.softmax(X, dim=1)
params_model={
    "shape_in": (3, 32, 32), 
    "initial_filters": 8,    
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2}

def build_binary(num_class,  use_pretrained=False):
    model = Network(params_model).cuda()
    return model

def build(num_class,  use_pretrained=False):
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = build_fn(hparams['arch'], num_class,use_pretrained)
    model = model.cuda()
    return model

def evaluate(dataloader,model):
    model = model.eval()
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

def train(train_loader,val_loader,num_class,use_pretrained=False,log_model=False,model_save_path=''):
    '''return the last checkpoint'''
    model = build(num_class,use_pretrained)
    training_args=hparams['training']
    training_args['optimizer'] = hparams['optimizer']
    training_args['iters_per_epoch'] = len(train_loader)
    trainer = trainer_utils.LightWeightTrainer(training_args=hparams['training'],
                                                    exp_name=model_save_path, enable_logging=log_model,
                                                    bce=False, set_device=True)
    _ = trainer.fit(model, train_loader, val_loader)
    return model # the last ckpnt  

def train_binary(train_loader,val_loader,num_class,use_pretrained=False,log_model=False,model_save_path=''):
    '''return the last checkpoint'''
    model = build_binary(num_class,use_pretrained)
    training_args=hparams['training']
    training_args['optimizer'] = hparams['optimizer']
    training_args['iters_per_epoch'] = len(train_loader)
    trainer = trainer_utils.LightWeightTrainer(training_args=hparams['training'],
                                                    exp_name=model_save_path, enable_logging=log_model,
                                                    bce=False, set_device=True)
    _ = trainer.fit(model, train_loader, val_loader)
    return model # the last ckpnt  

def tune(base_model,train_loader,val_loader, weights=None,log_model=False,model_save_path=''):
    training_args=hparams['tuning']
    training_args['iters_per_epoch'] = len(train_loader)
    training_args['optimizer'] = hparams['optimizer']
    trainer = trainer_utils.LightWeightTrainer(training_args=training_args,
                                                    exp_name=model_save_path, enable_logging=log_model,
                                                    bce=False, set_device=True,weights=weights)
    best_model = trainer.fit(base_model, train_loader, val_loader)
    return best_model # check point during refinig
    # return base_model
def get_new(new_model_config, train_loader, val_loader, old_model=None):
    if new_model_config.setter=='refine':
        new_model = tune(old_model,train_loader,val_loader) # tune
    else:
        new_model = train(train_loader,val_loader,num_class=new_model_config.class_number) # retrain 
    return new_model

def load_binary(model_config:OldModel,use_pretrained=False):
    # build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    out = torch.load(model_config.path,map_location=model_config.device)
    # model = build_fn(**out['build_params'],use_pretrained=use_pretrained) #test model or fine-tune model not feature extraction
    model = build(model_config.class_number,use_pretrained)
    model.load_state_dict(out['state_dict'])
    model = model.cuda()
    print('load model from {}'.format(model_config.path))
    # print(out['run_metadata'])
    return model

def save_binary(model, path):
    torch.save({
        # 'build_params': model._build_params,
        'state_dict': model.state_dict(),
    }, path)
    print('model saved to {}'.format(path))

def load(model_config:OldModel,use_pretrained=False):
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    out = torch.load(model_config.path,map_location=model_config.device)
    model = build_fn(**out['build_params'],use_pretrained=use_pretrained) #test model or fine-tune model not feature extraction
    model = build(model_config.class_number,use_pretrained)
    model.load_state_dict(out['state_dict'])
    model = model.cuda()
    print('load model from {}'.format(model_config.path))
    # print(out['run_metadata'])
    return model

def save(model, path):
    torch.save({
        'build_params': model._build_params,
        'state_dict': model.state_dict(),
    }, path)
    print('model saved to {}'.format(path))

def shift_importance(dataset, n_class, gt, pred):
    check_labels = config['data']['remove_fine_labels']
    check_labels_cnt = [0 for i in range(n_class)]
    for c in range(n_class):
        cls_mask = (gt==c)
        cls_incor_mask = (gt!=pred)[cls_mask]
        cls_idx = extract_class_indices(c, gt)
        cls_incor_idx = cls_idx[cls_incor_mask]
        for idx in cls_incor_idx:
            _, coarse_target, target = dataset[idx]
            if target in check_labels:
                check_labels_cnt[coarse_target] += 1             
        check_labels_cnt[c] = check_labels_cnt[c]/len(cls_incor_idx) * 100
    return check_labels_cnt
