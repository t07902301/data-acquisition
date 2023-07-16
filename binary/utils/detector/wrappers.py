import torch
import pickle as pkl
import clip
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import utils.detector.svm as svm_utils
import torch.nn as nn

def inv_norm(ds_mean, ds_std):
    if ds_std is None:
        return (lambda x: x)
    # invert normalization (useful for visualizing)    
    return transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [255/x for x in ds_std]),
                                transforms.Normalize(mean = [-x /255 for x in ds_mean],
                                                     std = [ 1., 1., 1. ]),
                               ])
class PreProcessing(nn.Module):
    
    def __init__(self, do_normalize=False, do_standardize=True):
        super().__init__()
        self.do_normalize = do_normalize
        self.do_standardize = do_standardize
        
    def update_stats(self, latents):
        # latents = latents.detach().clone()
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        self.mean = latents.mean(dim=0)
        self.std = (latents - self.mean).std(dim=0)
        self.max = latents.max()
        self.min = latents.min()
    
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
            print('Not tensors in detector input processing')
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
            'max': self.max
        }
    
    def _import(self, args):
        self.mean = args['mean']
        self.std = args['std']
        self.do_normalize = args['normalize']

from abc import abstractmethod

class Prototype:
    def __init__(self, do_normalize, do_standardize) -> None:
        self.pre_process = None
        self.do_normalize = do_normalize
        self.do_standardize = do_standardize
        self.clf = None

    @abstractmethod
    def export(self, filename):
        pass
    
    @abstractmethod
    def import_model(self, filename):
        pass
        
    def set_preprocess(self, train_latents=None):
        self.pre_process = PreProcessing(do_normalize=self.do_normalize)
        if train_latents is not None:
            print("updating whitening")
            self.pre_process.update_stats(train_latents)
        else:
            print("No whitening")

    @ abstractmethod
    def fit(self, latents, gts):
        pass

    def fit_preprocess(self, latents):
        assert self.pre_process is not None, 'run set_preprocess on a training set first'
        latents = self.pre_process(latents).numpy()
        return latents

    @abstractmethod
    def predict(self, latents):
        pass

    def predict_preprocess(self, latents):
        assert self.clf is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        return latents

from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
class LogRegressor(Prototype):
    def __init__(self, do_normalize=False, do_standardize=True) -> None:
        super().__init__(do_normalize, do_standardize)

    def fit(self, latents, gts):
        latents = self.fit_preprocess(latents)
        self.clf = LogisticRegression(random_state=0, max_iter=50, solver='liblinear')
        self.clf.fit(latents, gts)
    
    def predict(self, latents, gts=None, compute_metrics=False):
        latents = self.predict_preprocess(latents)
        dv = self.clf.decision_function(latents)
        preds = self.clf.predict(latents)
        metrics = None
        if compute_metrics and (gts is not None):
            metrics = balanced_accuracy_score(gts, preds) * 100
        return dv, metrics  
    
    def import_model(self, filename):
        with open(filename, 'rb') as f:
            out = pkl.load(f)
        self.clf = out['clf']
        # svm_stats = out['pre_stats']
        self.pre_process = PreProcessing(do_normalize=self.do_normalize, do_standardize=self.do_standardize)

    def export(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump({
                'clf': self.clf,
                'pre_stats': self.pre_process._export(),
                }, 
                f
            )

class SVM(Prototype):
    def __init__(self, split_and_search=True, balanced=True, cv=2, do_normalize=False, args=None, do_standardize=True):
        super().__init__(do_normalize, do_standardize)
        self.split_and_search = split_and_search
        self.balanced = balanced
        self.cv = cv
        self.clf = None
        self.args = args

    def fit(self, latents, gts):
        assert self.pre_process is not None, 'run set_preprocess on a training set first'
        latents = self.pre_process(latents).numpy()
        clf, score = svm_utils.train(latents, gts, balanced=self.balanced, 
                                    split_and_search=self.split_and_search,args=self.args)
        self.clf = clf
        return score
    
    def predict(self, latents, gts=None, compute_metrics=False):
        assert self.clf is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        return svm_utils.predict(self.clf, latents, gts, compute_metrics) 

    def base_fit(self, gts, latents):
        assert self.pre_process is not None, 'run set_preprocess on a training set first'
        latents = self.pre_process(latents).numpy()
        clf, score = svm_utils.base_train(latents=latents, gts=gts, balanced=self.balanced, 
                                                        split_and_search=self.split_and_search,args=self.args)
        self.clf = clf
        return score
    
    def base_predict(self, latents):
        assert self.clf is not None, "must call fit first"
        latents = self.pre_process(latents).numpy()
        pred = svm_utils.base_predict(latents, self.clf) 
        return pred

    def export(self, filename):
        args = {
            'split_and_search': self.split_and_search,
            'balanced': self.balanced,
            'cv': self.cv,
            'do_normalize': self.do_normalize,
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
        self.split_and_search=out['args']['split_and_search']
        self.balanced = out['args']['balanced']
        self.cv = out['args']['cv']
        self.do_normalize = out['args']['do_normalize']
        self.pre_process = PreProcessing(do_normalize=self.do_normalize, do_standardize=self.do_standardize)

    def legacy_import_model(self, filename, svm_stats):
        with open(filename, 'rb') as f:
            self.clf = pkl.load(f)
        self.pre_process = PreProcessing(do_normalize=True,
                                                      mean=svm_stats['mean'],
                                                      std=svm_stats['std'])
        
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
    
    def evaluate_clip_captions(self, captions):
        text = clip.tokenize(captions)
        ds = torch.utils.data.TensorDataset(text)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, drop_last=False, shuffle=False)
        clip_activations = []
        with torch.no_grad():
            for batch in dl:
                caption = batch[0].cuda()
                text_features = self.clip_model.encode_text(caption)
                clip_activations.append(text_features.cpu())
        return torch.cat(clip_activations).float()
   
    def get_caption_scores(self, captions, reference_caption, svm_fitter, target_c):
        caption_latent = self.evaluate_clip_captions(captions)
        reference_latent = self.evaluate_clip_captions([reference_caption])[0]
        latent = caption_latent - reference_latent
        gts = (torch.ones(len(latent))*target_c).long()
        _, decisions = svm_fitter.predict(gts=gts, latents=latent, compute_metrics=False)
        return decisions, caption_latent