import matplotlib.pyplot as plt
import utils.objects.Detector as Detector
import utils.objects.model as Model
import utils.objects.Config as Config
import os
import numpy as np

from scipy.stats import norm, gaussian_kde, kstest
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

def get_overlap_area(lb, ub, arr):
    lb_mask = (arr>lb)
    arr_lb = arr[lb_mask]
    ub_mask = (arr_lb<ub)
    arr_lb_ub = arr_lb[ub_mask]
    return arr_lb_ub
    
def get_kde(values):
    kernel = gaussian_kde(values)
    return kernel

def ecdf(raw_values, test_value=None):
    intervals = get_val_space(raw_values)
    raw_values = np.array(raw_values)
    n_raw_vals = len(raw_values)
    result = []
    if test_value == None:
        for edge in intervals:
            result.append(np.sum(raw_values <= edge) / n_raw_vals)
        return result
    else:
        return np.array([np.sum(raw_values <= test_value) / n_raw_vals])
    
def plot(values, n_bins, path, clf_metrics, removal_ratio):
    val_key = 'correct pred'
    bin_value, bins, _ = plt.hist(values[val_key], 
           bins=n_bins, alpha=0.3, density=True, color='orange')
    # pdf_x = get_val_space(values[val_key])
    # norm_pdf = get_norm_pdf(values[val_key])
    # plt.plot(pdf_x, norm_pdf.cdf(pdf_x), label=val_key, color='orange')
    # # cor_area = norm_pdf.cdf([0])
    # plt.plot(pdf_x, ecdf(values[val_key]), label=val_key, color='red')
    cor_area = ecdf(values[val_key], 0)

    # ks_result = kstest(values[val_key], norm_pdf.cdf)
    # print(ks_result.statistic, ks_result.pvalue)

    # kde = get_kde(values[val_key])
    # plt.plot(pdf_x, kde.evaluate(pdf_x), label=val_key, color='orange')

    val_key = 'incorrect pred'
    plt.hist(values[val_key], 
            bins=n_bins, alpha=0.3, density=True, color='blue')
    # pdf_x = get_val_space(values[val_key])
    # norm_pdf = get_norm_pdf(values[val_key])
    # plt.plot(pdf_x, norm_pdf.cdf(pdf_x), label=val_key, color='blue')
    # # incor_area = 1 - norm_pdf.cdf([0])
    # plt.plot(pdf_x, ecdf(values[val_key]), label=val_key, color='purple')
    incor_area = 1 - ecdf(values[val_key], 0)

    # ks_result = kstest(values[val_key], norm_pdf.cdf)
    # print(ks_result.statistic, ks_result.pvalue)

    # kde = get_kde(values[val_key])
    # plt.plot(pdf_x, kde.evaluate(pdf_x), label=val_key, color='blue')

    plt.xlabel('decision values')
    plt.ylabel('density')
    format_metrics = 'old_data: {}%; '.format((1-removal_ratio)*100)
    plt.title(format_metrics)
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()
    print('save fig to', path)   
    return incor_area + cor_area
    # return 0

def get_fig_name(fig_dir, model_type, model_cnt, removal_ratio):
    # fig_root = 'figure/{}/stat_test/{}'.format(fig_dir, model_type)
    fig_root = 'figure/{}/stat_test/{}/overlap/{}'.format(fig_dir, model_type, model_cnt)

    if os.path.exists(fig_root) is False:
        os.makedirs(fig_root)    
    fig_path = os.path.join(fig_root, '{}.png'.format(removal_ratio))
    return fig_path

def run(clf:Detector.SVM, dataloader, base_model: Model.prototype, model_config: Config.OldModel, removal_ratio=0):
    dataset_gts, dataset_preds, _ = base_model.eval(dataloader)
    dv, precision = clf.predict(dataloader, compute_metrics=True, base_model=base_model)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    cor_dv = dv[cor_mask]
    incor_dv = dv[incor_mask]
    fig_path = get_fig_name(model_config.model_dir, model_config.base_type, model_config.model_cnt, removal_ratio)
    total_dv = {
        'correct pred': cor_dv,
        'incorrect pred': incor_dv
    }
    clf_metrics = {
        'SVM ': np.round(precision, decimals=2),
        'Model': np.round(cor_mask.mean()*100, decimals=2)
    }
    intersection_area = plot(total_dv, n_bins=10, path=fig_path, clf_metrics=clf_metrics, removal_ratio = removal_ratio)
    return intersection_area
    # print(stat(total_dv['correct pred'], total_dv[val_key]))
