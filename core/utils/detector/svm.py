import numpy as np
# from sklearn import svm
from sklearn.svm import LinearSVC, SVC
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train(latents, gts, balanced=True, split_and_search=False, cv=2, args=None):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    class_weight = 'balanced' if balanced else None        
    kernel = args['kernel']
    best_clf, best_cv = choose_svm_hpara(latents, gts, class_weight, cv, kernel, split_and_search)
    best_clf.fit(latents, gts)
    return best_clf, best_cv  

def shuffl_train(latents, model_gts, model_preds, balanced=True, split_and_search=False, cv=2, args=None, C_ =1):
    print(C_)
    class_weight = 'balanced' if balanced else None        
    model_correct_pred_mask = (model_preds == model_gts)
    model_correctness = np.zeros(len(model_gts))
    model_correctness[model_correct_pred_mask] = 1
    kernel = args['kernel']
    best_clf = SVC(C= C_, kernel=kernel, class_weight=class_weight, gamma='auto')
    best_clf.fit(latents, model_correctness)
    return best_clf, None  

def choose_svm_hpara(clf_input, gt, class_weight, cv_splits, kernel, split_and_search):
    '''
    select the best hparameter for svm by cross validation
    '''
    best_C, best_cv, best_clf = 1, -np.inf, None
    if split_and_search:
        print('Grid Search')
        for C_ in np.logspace(-6, 0, 7, endpoint=True):
            # x_resampled, gt_resampled = SMOTE().fit_resample(x, gt)
            # clf, cv_score = fit_svm(C=C_, x=x_resampled, gt=gt_resampled, class_weight=class_weight, cv=cv, kernel=kernel)
            clf, cv_score = fit_svm(C=C_, x=clf_input, gt=gt, class_weight=class_weight, cv_splits=cv_splits, kernel=kernel)
            if cv_score > best_cv:
                best_cv = cv_score
                best_C = C_
                best_clf = clf
        print('best C:', best_C)
    else:
        best_clf = SVC(C=1, kernel=kernel, class_weight=class_weight, gamma='auto')
    return best_clf, best_cv

def fit_svm(C, class_weight, x, gt, cv_splits=2, kernel='linear'):
    '''
    x : input of svm; gt: groud truth of svm; cv: #cross validation splits
    '''
    cv = StratifiedKFold(shuffle=True, random_state=0, n_splits=cv_splits) # randomness in shuffling for cross validation
    # clf = LinearSVC(C=C, class_weight=class_weight, random_state=0)
    clf = SVC(C=C, kernel=kernel, class_weight=class_weight, gamma='auto', random_state=0) # randomness in shuffling for svm training
    scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
    average_cv_scores = np.mean(cv_scores)*100
    # print('cv scores',cv_scores)
    return clf, average_cv_scores

def predict(clf: SVC, latents, gts=None, compute_metrics=False):
    dataset_size = len(latents)
    out_mask, out_decision = np.zeros(dataset_size), np.zeros(dataset_size)
    out_decision = clf.decision_function(latents)
    metric = None
    if compute_metrics:
        preds = clf.predict(latents)
        metric = sklearn_metrics.balanced_accuracy_score(gts, preds)*100
    return out_mask, out_decision, metric


def base_train(latents, gts, balanced=True, split_and_search=False, cv=2, args=None):
    class_weight = 'balanced' if balanced else None        
    kernel = args['kernel']
    best_clf, best_cv = choose_svm_hpara(latents, gts, class_weight, cv, kernel, split_and_search)
    best_clf.fit(latents, gts)
    return best_clf, best_cv  

def raw_predict(latents, clf: SVC):
    return clf.predict(latents)