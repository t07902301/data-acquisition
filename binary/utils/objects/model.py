import sys
from binary.utils.objects.Config import OldModel
sys.path.append('..')
import failure_directions.src.model_utils as model_utils
import failure_directions.src.trainer as trainer_utils
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from utils.env import model_env
from utils import config
from utils.objects.Config import OldModel
from utils.detector.wrappers import SVMFitter, CLIPProcessor
from abc import abstractmethod

model_env()
hparams = config['hparams']
clip_processor = CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std'])

class prototype():
    @abstractmethod
    def __init__(self) -> None:
        pass
    @abstractmethod
    def eval(self, dataloader):
        '''
        gts, preds, conf(not from SVM for now)
        '''
        pass
    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def save(self, path):
        pass
    @abstractmethod
    def update(self):
        pass
    @abstractmethod
    def load(self, model_config:OldModel):
        pass

class resnet(prototype):
    def __init__(self, num_class,  use_pretrained=False) -> None:
        super().__init__()
        build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
        model = build_fn(hparams['arch'], num_class,use_pretrained)
        model = model.cuda()
        self.model = model

    def load(self, model_config: OldModel, use_pretrained=False):
        super().load(model_config)
        out = torch.load(model_config.path,map_location=model_config.device)
        build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
        model = build_fn(**out['build_params'],use_pretrained=use_pretrained) #test model or fine-tune model not feature extraction
        model.load_state_dict(out['state_dict'])
        print('load model from {}'.format(model_config.path))

    def eval(self, dataloader):
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

    def train(self, train_loader,val_loader,log_model=False,model_save_path=''):
        '''return the last checkpoint'''
        training_args=hparams['training']
        training_args['optimizer'] = hparams['optimizer']
        training_args['iters_per_epoch'] = len(train_loader)
        trainer = trainer_utils.LightWeightTrainer(training_args=hparams['training'],
                                                        exp_name=model_save_path, enable_logging=log_model,
                                                        bce=False, set_device=True)
        best_model_chkpnt = trainer.fit(self.model, train_loader, val_loader)
        self.model = best_model_chkpnt

    def tune(self,train_loader,val_loader, weights=None,log_model=False,model_save_path=''):
        training_args=hparams['tuning']
        training_args['iters_per_epoch'] = len(train_loader)
        training_args['optimizer'] = hparams['optimizer']
        trainer = trainer_utils.LightWeightTrainer(training_args=training_args,
                                                        exp_name=model_save_path, enable_logging=log_model,
                                                        bce=False, set_device=True,weights=weights)
        best_model_chkpnt = trainer.fit(self.model, train_loader, val_loader)
        self.model = best_model_chkpnt

    def save(self, path):
        torch.save({
            'build_params': self.model._build_params,
            'state_dict': self.model.state_dict(),
        }, path)
        print('model saved to {}'.format(path))
    
    def update(self, new_model_config, train_loader, val_loader, old_model=None):
        if new_model_config.setter=='refine':
            self.tune(old_model,train_loader,val_loader) # tune
        else:
            self.train(train_loader,val_loader,num_class=new_model_config.class_number) # retrain 

class svm(prototype):
    def __init__(self, set_up_dataloader, split_and_search=False) -> None:
        super().__init__()
        set_up_embedding, _ = clip_processor.evaluate_clip_images(set_up_dataloader)        
        self.model = SVMFitter(method=config['clf'], svm_args=config['clf_args'],cv=config['clf_args']['k-fold'], split_and_search = split_and_search)
        self.model.set_preprocess(set_up_embedding) 
    
    def load(self, model_config: OldModel):
        super().load(model_config)
        self.model.clf = torch.load(model_config.path)

    def eval(self, dataloader):
        fit_embedding, gts = clip_processor.evaluate_clip_images(dataloader)
        preds = self.model.base_predict(fit_embedding)
        return gts, preds, None
        
    def train(self, train_loader):
        embedding, gts = clip_processor.evaluate_clip_images(train_loader)        
        _ = self.model.base_fit(gts, embedding)

    def save(self, path):
        torch.save(self.model.clf, path)
        print('model saved to {}'.format(path))
    
    def update(self,train_loader):
        self.train(train_loader)
