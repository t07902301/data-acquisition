import torch
import pickle as pkl
import clip
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import torch.nn as nn
from utils.logging import *
import utils.detector.learner as Learner
def inv_norm(ds_mean, ds_std):
    if ds_std is None:
        return (lambda x: x)
    return transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/x for x in ds_std]),
                                transforms.Normalize(mean = [-x for x in ds_mean],
                                                     std = [ 1., 1., 1. ]),
                               ])
class PreProcessing(nn.Module):
    
    def __init__(self, do_normalize=False, do_standardize=True):
        super().__init__()
        self.do_normalize = do_normalize
        self.do_standardize = do_standardize
        
    def update_stats(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        self.mean = latents.mean(dim=0)
        self.std = (latents - self.mean).std(dim=0)
        self.max = latents.max(dim=0)
        self.min = latents.min(dim=0)
    
    def normalize(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        # return latents/torch.linalg.norm(latents, dim=1, keepdims=True)
        return (latents - self.min) / (self.max - self.min)
    
    def standardize(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        return (latents - self.mean) / self.std
    
    def forward(self, latents):
        if not torch.is_tensor(latents):
            logger.info('Not tensors in detector input processing')
            # latents = torch.tensor(latents) 
            latents = torch.concat(latents)  
        if self.do_standardize:
            latents = self.standardize(latents)
        if self.do_normalize:
            latents = self.normalize(latents)
        return latents
    
    def _export(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'do_normalize': self.do_normalize,
            'do_standardize': self.do_standardize
        }
    
    def _import(self, stats):
        self.mean = stats['mean']
        self.std = stats['std']
        self.do_normalize = stats['do_normalize']
        self.do_standardize = stats['do_standardize']
        self.max = stats['max']
        self.min = stats['min']

class Prototype:
    def __init__(self, split_and_search=True, balanced=True, cv=2, do_normalize=False, args=None, do_standardize=True) -> None:
        self.pre_process = None
        self.do_normalize = do_normalize
        self.do_standardize = do_standardize
        self.clf = None
        self.split_and_search = split_and_search
        self.balanced = balanced
        self.cv = cv
        self.args = args

        self.learner = Learner.Prototype()
        
    def export(self, filename):
        args = {
            'split_and_search': self.split_and_search,
            'balanced': self.balanced,
            'cv': self.cv,
        }
        with open(filename, 'wb') as f:
            pkl.dump({
                'clf': self.clf,
                'pre_stats': self.pre_process._export(),
                'args': args}, 
                f
            )
    def import_model(self, filename):
        with open(filename, 'rb') as f:
            out = pkl.load(f)
        self.clf = out['clf']
        self.pre_process = PreProcessing()
        self.pre_process._import(out['pre_stats'])

        self.split_and_search=out['args']['split_and_search']
        self.balanced = out['args']['balanced']
        self.cv = out['args']['cv']
        
    def set_preprocess(self, train_latents=None):
        self.pre_process = PreProcessing(do_normalize=self.do_normalize)
        if train_latents is not None:
            self.pre_process.update_stats(train_latents)
        else:
            logger.info("No whitening")

    def fit(self, latents, gts):
        assert self.pre_process is not None, 'run set_preprocess on a training set first'
        latents = self.pre_process(latents).numpy()
        self.clf, score = self.learner.train(latents, gts, balanced=self.balanced, 
                                    split_and_search=self.split_and_search,args=self.args)
        return score
    
    def predict(self, latents, gts=None, metrics=None):
        assert self.clf is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        return self.learner.predict(self.clf, latents, gts, metrics) 
    
    def raw_predict(self, latents):
        assert self.clf is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        pred = self.learner.raw_predict(latents, self.clf) 
        return pred

class LogRegressor(Prototype):
    def __init__(self, split_and_search=True, balanced=True, cv=2, do_normalize=False, args=None, do_standardize=True) -> None:
        super().__init__(split_and_search, balanced, cv, do_normalize, args, do_standardize)
        self.learner = Learner.logreg()

class SVM(Prototype):
    def __init__(self, split_and_search=True, balanced=True, cv=2, do_normalize=False, args=None, do_standardize=True) -> None:
        super().__init__(split_and_search, balanced, cv, do_normalize, args, do_standardize)
        self.learner = Learner.svm()

class CLIPProcessor:
    def __init__(self, ds_mean=0, ds_std=1, 
                 arch='ViT-B/32', device='cuda'):
        self.clip_model, preprocess = clip.load(arch, device=device)
        self.clip_model = self.clip_model.eval()
        clip_normalize = preprocess.transforms[-1]
        self.preprocess_clip = transforms.Compose(
            [
                inv_norm(ds_mean, ds_std),
                transforms.Resize((224, 224)),
                clip_normalize,
            ]
        )
        self.device = device
        
    def evaluate_clip_images(self, dataloader):
        clip_activations = []
        clip_gts = []
        with torch.no_grad():
            with autocast():
                for batch_info in dataloader:
                    x = batch_info[0] # (image, coarse_label, fine_label)
                    x = x.to(self.device)
                    image_features = self.clip_model.encode_image(self.preprocess_clip(x))
                    clip_activations.append(image_features.cpu())
                    clip_gts.append(batch_info[1])
        out = torch.cat(clip_activations).float()
        clip_gts = torch.cat(clip_gts).numpy()
        return out, clip_gts
    