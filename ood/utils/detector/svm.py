import torch
import numpy as np
# from sklearn import svm
from sklearn.svm import LinearSVC, SVC
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score
import torch.nn as nn
import torch.nn as nn

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

def train(latents, model_gts, model_preds, balanced=True, split_and_search=False, cv=2, svm_args=None):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    class_weight = 'balanced' if balanced else None        
    model_correct_pred_mask = (model_preds == model_gts)
    model_predictions = np.zeros(len(model_gts))
    model_predictions[model_correct_pred_mask] = 1
    kernel = svm_args['kernel']
    best_clf, best_cv = choose_svm_hpara(latents, model_predictions, class_weight, cv, kernel, split_and_search)
    best_clf.fit(latents, model_predictions)
    return best_clf, best_cv  

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

def fit_svm(C, class_weight, x, gt, cv=2, kernel='linear'):
    '''
    x : input of svm; gt: groud truth of svm; cv: #cross validation splits
    '''
    scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    # clf = LinearSVC(C=C, class_weight=class_weight)
    # print(kernel)
    clf = SVC(C=C, kernel=kernel, class_weight=class_weight, gamma='auto')
    cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
    average_cv_scores = np.mean(cv_scores)*100
    return clf, average_cv_scores

def predict(latents, gts, clf, preds=None, compute_metrics=False):
    dataset_size = len(gts)
    out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
    dataset_idx = np.arange(dataset_size)
    real_incorrect_mask = (gts!=preds)
    precision = []
    out_decision = clf.decision_function(latents)
    
    if compute_metrics:
        clf_incorrect_mask = (out_decision<=0)
        incorrect_idx = dataset_idx[clf_incorrect_mask]
        real_incorrect_idx = dataset_idx[real_incorrect_mask]
        if  len(incorrect_idx) == 0:
            precision.append(100)
            print('No misclassifications')
        else:
            precision.append(np.intersect1d(incorrect_idx, real_incorrect_idx).size / incorrect_idx.size * 100)  

    return out_mask, out_decision, precision

def base_train(latents, gts, balanced=True, split_and_search=False, cv=2, svm_args=None):
    class_weight = 'balanced' if balanced else None        
    kernel = svm_args['kernel']
    best_clf, best_cv = choose_svm_hpara(latents, gts, class_weight, cv, kernel, split_and_search)
    best_clf.fit(latents, gts)
    return best_clf, best_cv  

def base_predict(latents, gts, clf):
    return clf.score(latents, gts)