from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import LinearSVC, SVC
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.logging import *
from utils.env import seed, data_split_env
from abc import abstractmethod

class Prototype():
    def __init__(self) -> None:
        pass

    def train(self, latents, gts, balanced=True, split_and_search=False, cv=2, args=None):
        # if split_and_search is true, split our dataset into 2-folds (50:50)
        # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
        class_weight = 'balanced' if balanced else None        
        best_clf, best_cv = self.choose_hparam(latents, gts, class_weight, cv, split_and_search, args)
        best_clf.fit(latents, gts)
        return best_clf, best_cv  

    def choose_hparam(self, clf_input, gt, class_weight, cv_splits, split_and_search, args):
        '''
        select the best hparameter for svm by cross validation
        '''
        best_C, best_cv, best_clf = 1, -np.inf, None
        if split_and_search:
            logger.info('Grid Search')
            for C_ in np.logspace(-6, 0, 7, endpoint=True):
                # x_resampled, gt_resampled = SMOTE().fit_resample(x, gt)
                clf, cv_score = self.fit(C=C_, x=clf_input, gt=gt, class_weight=class_weight, cv_splits=cv_splits, args=args)
                if cv_score > best_cv:
                    best_cv = cv_score
                    best_C = C_
                    best_clf = clf
            logger.info('best C:{}'.format(best_C))
        else:
            best_clf, best_cv = self.fit(C=best_C, x=clf_input, gt=gt, class_weight=class_weight, cv_splits=cv_splits, args=args)
        return best_clf, best_cv
    
    @abstractmethod
    def fit(self, C, class_weight, x, gt, cv_splits=2, args= None):
        pass

    @abstractmethod
    def predict(self, clf, latents, gts=None, metrics=None):
        '''
        Feature Scores from class 0 (gt!=pred) \n
        SVM : opposite decision values \n
        Logistic Regression: prediction probability
        '''
        pass
    
    def raw_predict(self, latents, clf):
        return clf.predict(latents)
    
    def compute_metrics(self, clf, latents, gts, metric='precision'):
        preds = clf.predict(latents)
        if metric == 'f1-score':
            metric = sklearn_metrics.f1_score(gts, preds, pos_label=0) * 100
        elif metric == 'recall':
            metric = sklearn_metrics.balanced_accuracy_score(gts, preds) * 100
        else:
            metric = sklearn_metrics.precision_score(gts, preds, pos_label=0) * 100
        return metric
    
class logreg(Prototype):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, C, class_weight, x, gt, cv_splits=2, args= None):
        '''
        x : input of svm; gt: groud truth of svm; cv: #cross validation splits
        '''
        cv = StratifiedKFold(shuffle=True, random_state=0, n_splits=cv_splits) # randomness in shuffling for cross validation
        clf = LogisticRegression(random_state=0, max_iter=50, solver='liblinear', C=C, class_weight=class_weight)
        scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
        cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
        average_cv_scores = np.mean(cv_scores)*100
        return clf, average_cv_scores
    
    def predict(self, clf: LogisticRegression, latents, gts=None, metrics=None):
        dataset_size = len(latents)
        out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
        out_decision = clf.predict_proba(latents)[:, 0]
        metric = None
        if metrics != None:
            metric = self.compute_metrics(clf, latents, gts, metrics)
        return out_mask, out_decision, metric
    
class svm(Prototype):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, C, class_weight, x, gt, cv_splits=2, args=None):
        data_split_env()
        kernel = args['kernel']
        cv = StratifiedKFold(shuffle=True, random_state=seed, n_splits=cv_splits) # randomness in shuffling for cross validation
        if kernel == 'linearSVC':
            clf = LinearSVC(C=C, class_weight=class_weight, random_state=seed, dual='auto')
        else:
            clf = SVC(C=C, kernel=kernel, class_weight=class_weight, gamma='auto', random_state=seed) # randomness in shuffling for svm training
        scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
        cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
        average_cv_scores = np.mean(cv_scores)*100
        return clf, average_cv_scores
    
    def predict(self, clf, latents, gts=None, metrics=None):
        dataset_size = len(latents)
        out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
        out_decision = -clf.decision_function(latents)
        metric = None
        if metrics != None:
            metric = self.compute_metrics(clf, latents, gts, metrics)
        return out_mask, out_decision, metric