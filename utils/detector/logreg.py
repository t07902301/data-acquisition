import numpy as np
from sklearn.linear_model import LogisticRegression
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train(latents, gts, balanced=True, split_and_search=False, cv=2):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    class_weight = 'balanced' if balanced else None        
    best_clf, best_cv = choose_hparam(latents, gts, class_weight, cv, split_and_search)
    best_clf.fit(latents, gts)
    return best_clf, best_cv  

def choose_hparam(clf_input, gt, class_weight, cv_splits, split_and_search):
    '''
    select the best hparameter for svm by cross validation
    '''
    best_C, best_cv, best_clf = 1, -np.inf, None
    if split_and_search:
        print('Grid Search')
        for C_ in np.logspace(-6, 0, 7, endpoint=True):
            # x_resampled, gt_resampled = SMOTE().fit_resample(x, gt)
            clf, cv_score = fit(C=C_, x=clf_input, gt=gt, class_weight=class_weight, cv_splits=cv_splits)
            if cv_score > best_cv:
                best_cv = cv_score
                best_C = C_
                best_clf = clf
        print('best C:', best_C)
    else:
        best_clf = LogisticRegression(random_state=0, max_iter=50, solver='liblinear', class_weight=class_weight)
    return best_clf, best_cv

def fit(C, class_weight, x, gt, cv_splits=2):
    '''
    x : input of svm; gt: groud truth of svm; cv: #cross validation splits
    '''
    cv = StratifiedKFold(shuffle=True, random_state=0, n_splits=cv_splits) # randomness in shuffling for cross validation
    clf = LogisticRegression(random_state=0, max_iter=50, solver='liblinear', C=C, class_weight=class_weight)
    scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
    average_cv_scores = np.mean(cv_scores)*100
    return clf, average_cv_scores

def predict(clf: LogisticRegression, latents, gts=None, compute_metrics=False):
    dataset_size = len(latents)
    out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
    out_decision = clf.decision_function(latents)
    metric = None
    if compute_metrics:
        preds = clf.predict(latents)
        metric = sklearn_metrics.balanced_accuracy_score(gts, preds)*100
    return out_mask, out_decision, metric


def raw_predict(latents, clf: LogisticRegression):
    return clf.predict(latents)