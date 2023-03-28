import sys
sys.path.append('..')
import failure_directions.src.model_utils as model_utils
import failure_directions.src.trainer as trainer_utils
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from utils.env import model_env
from utils import config
from utils.Config import OldModelConfig
model_env()
hparams = config['hparams']
def build_model(num_class,  use_pretrained=False):
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = build_fn(hparams['arch'], num_class,use_pretrained)
    model = model.cuda()
    return model
def evaluate_model(dataloader,model):
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
def train_model(train_loader,val_loader,num_class,use_pretrained=False,log_model=False,model_save_path=''):
    '''return the last checkpoint'''
    model = build_model(num_class,use_pretrained)
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
def get_new_model(new_model_config, train_loader, val_loader, old_model=None):
    if new_model_config.setter=='refine':
        new_model = tune(old_model,train_loader,val_loader) # tune
    else:
        new_model = train_model(train_loader,val_loader,num_class=new_model_config.class_number) # retrain 
    return new_model
def load_model(model_config:OldModelConfig,use_pretrained=False):
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    out = torch.load(model_config.path,map_location=model_config.device)
    model = build_fn(**out['build_params'],use_pretrained=use_pretrained) #test model or fine-tune model not feature extraction
    model.load_state_dict(out['state_dict'])
    model = model.cuda()
    print('load model from {}'.format(model_config.path))
    # print(out['run_metadata'])
    return model
def save_model(model, path):
    torch.save({
        'build_params': model._build_params,
        'state_dict': model.state_dict(),
    }, path)
    print('model saved to {}'.format(path))


