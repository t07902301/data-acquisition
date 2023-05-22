import matplotlib.pyplot as plt
import utils.objects.Detector as Detector
import utils.objects.model as Model
import utils.objects.Config as Config
import os
import numpy as np

from scipy.stats import norm
# from sklearn.neighbors import KernelDensity

def format_dict(input_dict:dict):
    output = ''
    for keys, values in input_dict.items():
        output += '{}: {}% '.format(keys, values)
    return output

def get_norm_pdf(values):
    mean = np.mean(values)
    std = np.std(values)
    dist = norm(mean, std)
    target = values
    uni_values = sorted(np.unique(target))
    probabilities = [dist.pdf(value) for value in uni_values]   
    return uni_values, probabilities

def get_overlap_bound(arr_1, arr_2):
    return max(np.min(arr_1),np.min(arr_2)), min(np.max(arr_1), np.max(arr_2))

def get_overlap_area(lb, ub, arr):
    lb_mask = (arr>lb)
    arr_lb = arr[lb_mask]
    ub_mask = (arr_lb<ub)
    arr_lb_ub = arr_lb[ub_mask]
    return arr_lb_ub
    
from scipy import stats
# def stat(arr_1, arr_2):
#     return stats.kstest(sample_1, smaple_2)

def plot(values, n_bins, path, clf_metrics):
    bin_value, bins, _ = plt.hist(values['correct pred'], 
           bins=n_bins, alpha=0.3, density=True, color='orange')
    pdf_x, pdf_y = get_norm_pdf(values['correct pred'])
    plt.plot(pdf_x, pdf_y, label='correct pred', color='orange')
    plt.hist(values['incorrect pred'], 
            bins=n_bins, alpha=0.3, density=True, color='blue')
    pdf_x, pdf_y = get_norm_pdf(values['incorrect pred'])
    plt.plot(pdf_x, pdf_y, label='incorrect pred', color='blue')
   
    format_metrics = format_dict(clf_metrics)
    plt.title(format_metrics)
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()
    print('save fig to', path)      

def get_fig_name(fig_dir, model_type, model_cnt):
    fig_root = 'figure/{}/stat_test/{}'.format(fig_dir, model_type)
    if os.path.exists(fig_root) is False:
        os.makedirs(fig_root)    
    fig_path = os.path.join(fig_root, '{}.png'.format(model_cnt))
    return fig_path

def run(clf:Detector.SVM, dataloader, base_model: Model.prototype, model_config: Config.OldModel):
    dataset_gts, dataset_preds, _ = base_model.eval(dataloader)
    dv, precision = clf.predict(dataloader, compute_metrics=True, base_model=base_model)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    cor_dv = dv[cor_mask]
    incor_dv = dv[incor_mask]
    fig_path = get_fig_name(model_config.model_dir, model_config.model_type, model_config.model_cnt)
    total_dv = {
        'correct pred': cor_dv,
        'incorrect pred': incor_dv
    }
    clf_metrics = {
        'classifier precision': np.round(precision, decimals=2),
        'base model val_acc': np.round(cor_mask.mean()*100, decimals=2)
    }
    # plot(total_dv, n_bins=10, path=fig_path, clf_metrics=clf_metrics)
    print(stat(total_dv['correct pred'], total_dv['incorrect pred']))
