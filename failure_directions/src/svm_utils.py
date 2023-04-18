import yaml
import sys
import torch
from tqdm import tqdm
import os
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score
import torch.nn as nn
import torch.optim as optim
import torchvision
import failure_directions.src.trainer as trainer_utils
import failure_directions.src.ffcv_utils as ffcv_utils
import torch.nn as nn
from torch.cuda.amp import autocast
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
# from src.trainer import SVMTrainer
import clip
import torchvision.transforms as transforms
import torchmetrics
import sklearn.neural_network
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE,ADASYN

class SVMPreProcessing(nn.Module):
    
    def __init__(self, mean=None, std=None, do_normalize=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.do_normalize = do_normalize
        
    def update_stats(self, latents):
        # latents = latents.detach().clone()
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        self.mean = latents.mean(dim=0)
        self.std = (latents - self.mean).std(dim=0)
    
    def normalize(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        return latents/torch.linalg.norm(latents, dim=1, keepdims=True)
    
    def whiten(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        return (latents - self.mean) / self.std
    
    def forward(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)    
        if self.mean is not None:
            latents = self.whiten(latents)
        if self.do_normalize:
            latents = self.normalize(latents)
        return latents
    
    def _export(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'normalize': self.do_normalize
        }
    
    def _import(self, args):
        self.mean = args['mean']
        self.std = args['std']
        self.do_normalize = args['normalize']
            
    

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        #self.inception_network = torchvision.models.inception_v3(pretrained=True)
        self.inception_network = torchvision.models.inception_v3(pretrained=False)
        self.inception_network.load_state_dict(torch.load(torch.hub.get_dir() + '/checkpoints/inception_v3_google-1a9a5a14.pth'))
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        # x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations



def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
      
    
def sweep_f1_scores(logits, y):
    thresholds = np.arange(0.01, 1, 0.01)
    f1_scores = []
    for threshold in thresholds:
        f1 = torchmetrics.F1Score(threshold=threshold)
        f1_scores.append(f1(logits, y.int()).item())
    print(thresholds)
    print(f1_scores)
    return thresholds[np.argmax(f1_scores)]
        
    
def evaluate_bce_loader(dl, model, class_index, set_device=False, bce_threshold=None):
    latents = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    def print_out(m, x_):
        latents.append(x_[0].cpu())
    handle = getattr(model, model._last_layer_str).register_forward_pre_hook(print_out)
    
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        with autocast():
            all_logits, ys, spuriouses, indices = [], [], [], []
            for batch in dl:
                x, y, spurious, idx = trainer_utils.unwrap_batch(batch, set_device=set_device)
                y = y[:, class_index]
                if len(y) != len(x):
                    x = x[:len(y)] # for drop_last
                out = model(x)
                logits = sigmoid(out)[:, class_index] 
                all_logits.append(logits.cpu())
                ys.append(y.cpu())
                if spurious is not None:
                    spuriouses.append(spurious.cpu())
                else:
                    spuriouses.append(None)
                indices.append(idx.cpu())
    all_logits = torch.cat(all_logits)
    ys = torch.cat(ys)
    
    if bce_threshold is None:
        bce_threshold = sweep_f1_scores(all_logits, ys)
        print(bce_threshold)
        
    preds = (all_logits > bce_threshold).int().cpu()
    confs = torch.zeros_like(all_logits)
    confs[ys==1] = all_logits[ys==1]
    confs[ys==0] = 1-all_logits[ys==0]

    latents = torch.cat(latents)
    if spuriouses[0] is not None:
        spuriouses = torch.cat(spuriouses)
    else:
        spuriouses = None
    indices = torch.cat(indices)
    print("Accuracy", (preds == ys).float().mean().item())
    handle.remove()
    return {
        'preds': preds, 
        'ys': ys, 
        'spuriouses': spuriouses, 
        'latents': latents,
        'confs': confs,
        'indices': indices,
    }, bce_threshold

def evaluate_loader(dl, model, robust_model=False, set_device=False, save_classes=0, save_pred_probs=False):
    latents = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def print_out(m, x_):
        latents.append(x_[0].cpu())
    handle = getattr(model, model._last_layer_str).register_forward_pre_hook(print_out)
    with torch.no_grad():
        with autocast():
            preds, ys, spuriouses, confs, indices, pred_probs = [], [], [], [], [], []
            for batch in dl:
                x, y, spurious, idx = trainer_utils.unwrap_batch(batch, set_device=set_device)
                if len(y) != len(x):
                    x = x[:len(y)] # for drop_last
                out = model(x)
                softmax_logits = nn.Softmax(dim=-1)(out)
                if save_pred_probs:
                    pred_probs.append(softmax_logits.cpu())
                confs.append(softmax_logits[torch.arange(out.shape[0]), y].cpu())
                preds.append(out.argmax(-1).cpu())
                ys.append(y.cpu())
                if spurious is not None:
                    spuriouses.append(spurious.cpu())
                else:
                    spuriouses.append(None)
                indices.append(idx.cpu())
    if save_pred_probs:
        pred_probs = torch.cat(pred_probs)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    latents = torch.cat(latents)
    confs = torch.cat(confs)
    if spuriouses[0] is not None:
        spuriouses = torch.cat(spuriouses).cpu()
    else:
        spuriouses = None
    print("Accuracy", (preds == ys).float().mean().item())
    handle.remove()
    indices = torch.cat(indices)
    return {
        'preds': preds, 
        'ys': ys, 
        'spuriouses': spuriouses, 
        'latents': latents,
        'confs': confs,
        'indices': indices,
        'pred_probs': pred_probs,
    }

def evaluate_inception_features(dl, set_device=False):
    model = PartialInceptionNetwork()
    model = model.eval().cuda()
    resize = torchvision.transforms.Resize((299, 299))
    inception_activations = []
    with torch.no_grad():
        with autocast():
            for batch in dl:
                x, y, _, _ = trainer_utils.unwrap_batch(batch, set_device=set_device)
                if len(y) != len(x):
                    x = x[:len(y)]
                out = model(resize(x))
                inception_activations.append(out.cpu())
    return torch.cat(inception_activations)        

def get_accuracy(ytrue, ypred):
    if not torch.is_tensor(ytrue):
        ytrue = torch.tensor(ytrue)
    if not torch.is_tensor(ypred):
        ypred = torch.tensor(ypred)
    return float((ytrue == ypred).float().mean())*100

def clf_precision(ytrue, ypred):
    '''
    get average precision of SVM or other classifiers (clf)
    '''
    accs = []
    for c in range(2):
        cls_mask = ypred == c
        if cls_mask.sum()==0:
            accs.append(0)
        else:
            accs.append(get_accuracy(ytrue[cls_mask], ypred[cls_mask]))
    return np.mean(accs)

def clf_recall(ytrue, ypred):
    '''
    get recall of SVM or other classifiers (clf)
    '''
    accs = []
    for c in range(2):
        cls_mask = ytrue == c
        if cls_mask.sum()==0:
            accs.append(0)
        else:
            accs.append(get_accuracy(ytrue[cls_mask], ypred[cls_mask]))
    return np.mean(accs)

# =================================
# Per class SVM
# =================================

def fit_svm(C, class_weight, x, gt, cv=2, kernel='linear'):
    '''
    x : input of svm; gt: groud truth of svm; cv: #cross validation splits
    '''
    scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    clf = SVC(gamma='auto', kernel=kernel, C=C, class_weight=class_weight)
    cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
    average_cv_scores = np.mean(cv_scores)*100
    return clf, average_cv_scores
def choose_svm_hpara(clf_input, clf_gt, class_weight, cv_splits, kernel, split_and_search):
    '''
    select the best hparameter for svm by cross validation
    '''
    best_C, best_cv, best_clf = 1, -np.inf, None
    if split_and_search:
        for C_ in np.logspace(-6, 0, 7, endpoint=True):
            # x_resampled, clf_gt_resampled = SMOTE().fit_resample(x, clf_gt)
            # clf, cv_score = fit_svm(C=C_, x=x_resampled, gt=clf_gt_resampled, class_weight=class_weight, cv=cv, kernel=kernel)
            clf, cv_score = fit_svm(C=C_, x=clf_input, gt=clf_gt, class_weight=class_weight, cv=cv_splits, kernel=kernel)
            if cv_score > best_cv:
                best_cv = cv_score
                best_C = C_
                best_clf = clf
    else:
        # TODO
        # Set a default C_
        C_ = np.log(-6)
        best_clf, best_cv = fit_svm(C=C_, x=clf_input, gt=clf_gt, class_weight=class_weight, cv=cv_splits, kernel=kernel)
    return best_clf, best_cv

def train_per_class_svm(latents, gts, preds, balanced=True, split_and_search=False, cv=2, svm_args=None):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    class_num = gts.max() + 1
    class_weight = 'balanced' if balanced else None        
    val_correct_mask = (preds == gts)
    val_latent = latents
    clfs = []
    kernel = svm_args['kernel']
    cv_score = []
    for c in range(class_num):
        cls_mask = (gts == c)
        clf_input, clf_gt = val_latent[cls_mask], val_correct_mask[cls_mask]
        assert False in clf_gt, 'class {} only has correct classifications'.format(c)
        assert True in clf_gt, 'class {} only has misclassifications'.format(c)
        best_clf, best_cv = choose_svm_hpara(clf_input, clf_gt, class_weight, cv, kernel, split_and_search)
        best_clf.fit(clf_input, clf_gt)
        clfs.append(best_clf)
        cv_score.append(best_cv) # save cross valiadation score
    return clfs, cv_score
    
def train_per_class_model(latents, gts, preds, balanced=True, split_and_search=False, cv=2, method='SVM', svm_args={}):
    if method == 'SVM':
        clfs,score = train_per_class_svm(latents=latents, gts=gts, preds=preds, balanced=balanced, split_and_search=split_and_search, cv=cv,svm_args=svm_args)
    # else:
    #     clfs,score = train_per_class_mlp(latents=latents, ys=ys, preds=preds, balanced=balanced, split_and_search=split_and_search, cv=cv, svm_args=svm_args, market_info=market_info)
    return clfs,score

def predict_per_class_svm(latents, gts, clfs, preds=None, aux_info=None, verbose=True, compute_metrics=False, method='SVM'):
    dataset_size = len(gts)
    out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
    skipped_classes = []
    dataset_idx = np.arange(dataset_size)
    incorrect_mask = gts!=preds
    precision = []
    for c in range(len(clfs)): #replaced class_num
        cls_mask = gts == c
        if clfs[c] is not None and (cls_mask.sum()> 0):
            clf_out = clfs[c].predict(latents[cls_mask])
            if method == 'SVM':
                decision_out = clfs[c].decision_function(latents[cls_mask])
            else:
                decision_out = clfs[c].predict_proba(latents[cls_mask])[:,0]
            # out_mask[np.arange(dataset_size)[mask][clf_out == 1]] = 1
            out_decision[np.arange(dataset_size)[cls_mask]] = decision_out
            # score.append(get_accuracy(ytrue=correct_mask[cls_mask], ypred=clf_out)) # the base model and the clf prediction of correct classes
            # print('In class {}, confusion matrix'.format(c))
            # print(sklearn_metrics.confusion_matrix(correct[mask],clf_out))
            if compute_metrics:
                cls_idx = dataset_idx[cls_mask]
                clf_cls_incorrect_mask = (decision_out<=0)
                clf_cls_incorrect_idx = cls_idx[clf_cls_incorrect_mask]
                real_cls_incorrect_mask = incorrect_mask[cls_mask]
                real_cls_incorrect_idx = cls_idx[real_cls_incorrect_mask]
                if  clf_cls_incorrect_idx.size == 0:
                    precision.append(100)
                else:
                    precision.append(np.intersect1d(clf_cls_incorrect_idx, real_cls_incorrect_idx).size / clf_cls_incorrect_idx.size * 100)  
        else:
            skipped_classes.append(c)
    # ypred = out_mask.astype(int)
    # if compute_metrics:
    # #     clf_metrics = {k: np.array([u[k] for u in clf_metrics]) for k in clf_metrics[0].keys()}    
    # #     metric_dict = {
    # #         'accuracy': get_accuracy(ytrue=ytrue, ypred=ypred),
    # #         'balanced_accuracy': get_balanced_accuracy(ytrue=ytrue, ypred=ypred), 
    # #         'confusion_matrix': sklearn_metrics.confusion_matrix(ytrue, ypred),
    # #         'clf_accs': clf_metrics,
    # #         'ytrue': ytrue,
    # #         'ypred': ypred,
    # #         'decision_values': out_decision,
    # #         'classes': gts,
    # #         'skipped_classes': skipped_classes,
    # #         **aux_info
    # #     }
    # #     if verbose:
    # #         pprint(metric_dict, indent=4)
    #     return out_mask, out_decision, {}
    return out_mask, out_decision, precision


def predict_per_class_model(latents, gts, clfs, preds=None, aux_info=None, verbose=True, compute_metrics=True, method='SVM'):
    return predict_per_class_svm(latents=latents, gts=gts, clfs=clfs, preds=preds, aux_info=aux_info, verbose=verbose, compute_metrics=compute_metrics, method=method)

# def train_per_class_mlp(latents, ys, preds, balanced=True, split_and_search=False, cv=2, hidden_layer_size=100, svm_args={},market_info=None):
#     market_latents = market_info['latents']
#     market_ys = market_info['ys']
#     market_preds = market_info['preds']
#     market_correct = (market_preds==market_ys).type(torch.int32)

#     # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
#     # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
#     class_num = ys.max() + 1
#     #class_weight = 'balanced' if balanced else None        
#     val_correct = (preds == ys).type(torch.int32)
#     val_latent = latents
#     clfs = []
#     score = {'market_score': [], 'val_score':[],'cv':[]}
#     for c in range(class_num):
#         mask = ys == c
#         x, gt = val_latent[mask], val_correct[mask]
#         # ds_size = len(gt)
#         # np.random.shuffle(x)
#         # np.random.shuffle(gt)
#         # x_train,x_test = x[:int(ds_size*0.8)], x[int(ds_size*0.8):]
#         # gt_train,gt_test = gt[:int(ds_size*0.8)], gt[int(ds_size*0.8):]

#         # x_train,x_test = torch.utils.data.random_split( x ,[int(ds_size*0.8),int(ds_size*0.2)])
#         # gt_train,gt_test = torch.utils.data.random_split( gt ,[int(ds_size*0.8),int(ds_size*0.2)])
#         clf, cv_score = fit_mlp(x=x, gt=gt, svm_args=svm_args,cv=cv) 
#         score['cv'].append(cv_score)
#         score['val_score'].append(clf.score(x,gt))
        
#         market_mask = market_ys==c
#         market_x,market_gt = market_latents[market_mask],market_correct[market_mask]
#         score['market_score'].append(clf.score(market_x,market_gt)) 

#         clfs.append(clf)
#     return clfs, score


# def fit_mlp(x, gt, args={}):
#     '''
#     x : input of svm; gt: groud truth of mlp; args for mlp
#     '''    

#     #scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    
#     # make hidden layer based on feature dimension of x
#     # batch_size = first dimension
#     input_dim = x.shape[1]
#     hidden_layer_size = args['hidden_layer_size']
#     # if 'first_layer' in svm_args and not svm_args['first_layer']:
#     #     hidden_layer_sizes =(hidden_layer_size, )
#     # else:
#     #     hidden_layer_sizes=(input_dim, hidden_layer_size, )
#     hidden_layer_sizes = tuple(hidden_layer_size)
#     clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=args['max_iter'])
#     # scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
#     # cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
#     # average_cv_scores = np.mean(cv_scores)    
#     clf.fit(x, gt)
#     return clf, 0
