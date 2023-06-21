import matplotlib.pyplot as plt
import utils.objects.Detector as Detector
import utils.objects.model as Model
import utils.objects.Config as Config
import os
import numpy as np

from scipy.stats import norm, gaussian_kde, kstest, skewnorm, skew
# from sklearn.neighbors import KernelDensity

def format_dict(input_dict:dict):
    output = ''
    for keys, values in input_dict.items():
        output += '{}: {}%; '.format(keys, values)
    return output

def get_norm_pdf(values):
    mean = np.mean(values)
    std = np.std(values)
    dist = norm(mean, std)
    return dist

def get_val_space(values):
    max_val = max(values)
    min_val = min(values)
    val_space = np.linspace(min_val, max_val)   
    return val_space

def get_overlap_bound(arr_1, arr_2):
    return max(np.min(arr_1),np.min(arr_2)), min(np.max(arr_1), np.max(arr_2))

def get_kde(values):
    kernel = gaussian_kde(values)
    return kernel

def ecdf(raw_values, cut_value=None):
    raw_values = np.array(raw_values)
    n_raw_vals = len(raw_values)
    if cut_value == None:
        intervals = get_val_space(raw_values)
        result = []
        for edge in intervals:
            result.append(np.sum(raw_values <= edge) / n_raw_vals)
        return result
    else:
        return np.array([np.sum(raw_values <= cut_value) / n_raw_vals])
    
def get_overlap_area(values, cut_value):
    val_key = 'correct pred'
    cor_area = ecdf(values[val_key], cut_value)
    val_key = 'incorrect pred'
    incor_area = 1 - ecdf(values[val_key], cut_value)
    return incor_area + cor_area

def get_fig_name(fig_dir, model_type, model_cnt, removal_ratio):
    fig_root = 'figure/{}/stat_test/{}/overlap/{}'.format(fig_dir, model_type, model_cnt)
    if os.path.exists(fig_root) is False:
        os.makedirs(fig_root)    
    fig_path = os.path.join(fig_root, '{}.png'.format(removal_ratio))
    return fig_path

def get_dv_dstr(model: Model.prototype, dataloader, clf:Detector.Prototype):
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    dv, _ = clf.predict(dataloader, model)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    cor_dv = dv[cor_mask]
    incor_dv = dv[incor_mask]
    return cor_dv, incor_dv
    
def run(clf:Detector.SVM, dataloader, model_config: Config.OldModel, removal_ratio=0, n_bins = 20, plot=False, overlap_cut_value=None):
    model = Model.resnet(2)
    model.load(model_config)
    cor_dv, incor_dv = get_dv_dstr(model, dataloader, clf)
    total_dv = {
        'correct pred': cor_dv,
        'incorrect pred': incor_dv
    }

    # clf_metrics = {
    #     'SVM ': np.round(precision, decimals=2),
    #     'Model': np.round(cor_mask.mean()*100, decimals=2)
    # }
    # intersection_area = overlap_plot(total_dv, n_bins=n_bins, path=fig_path, clf_metrics=None, removal_ratio = removal_ratio)
    intersection_area = get_overlap_area(total_dv, overlap_cut_value)
    if plot:
        fig_path = get_fig_name(model_config.model_dir, model_config.base_type, model_config.model_cnt, removal_ratio)
        base_plot(total_dv['correct pred'], 'correct pred', 'orange', n_bins=n_bins)
        base_plot(total_dv['incorrect pred'], 'incorrect pred', 'blue', n_bins = n_bins)
        # format_metrics = 'old_data: {}%; '.format((1-removal_ratio)*100)
        # plt.title(format_metrics)
        plt.savefig(fig_path)
        plt.close()
        print('save fig to', fig_path)          
    return intersection_area

def base_plot(value, label, color, pdf_method=None, range=None, n_bins = 10):
    plt.hist(value, bins= n_bins , alpha=0.3, density=True, color=color, label=label, range=range)
    if pdf_method != None:
        dstr = get_pdf(value, pdf_method)
        pdf_x = get_val_space(value)
        if pdf_method == 'norm':
            plt.plot(pdf_x, dstr.pdf(pdf_x), color=color)
        else:
            plt.plot(pdf_x, dstr.evaluate(pdf_x), color=color)
    plt.legend()

def get_pdf(value, method):
    if method == 'norm':
        return get_norm_pdf(value)
    else:
        return get_kde(value)