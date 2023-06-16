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
    
def overlap_plot(values, n_bins, path, clf_metrics, removal_ratio):
    val_key = 'correct pred'
    bin_value, bins, _ = plt.hist(values[val_key], bins=n_bins, alpha=0.3, density=True, color='orange', label=val_key)
    pdf_x = get_val_space(values[val_key])
    # norm_pdf = get_norm_pdf(values[val_key])
    # plt.plot(pdf_x, norm_pdf.pdf(pdf_x), color='orange')
    # skewness = skew(values[val_key])
    # print('skewness:' , skewness)
    # plt.plot(pdf_x, skewnorm.pdf(pdf_x, skewness), color='orange')

    # cor_area = norm_pdf.cdf([0])
    # plt.plot(pdf_x, ecdf(values[val_key]), label=val_key, color='red')
    cor_area = ecdf(values[val_key], 0)
    print("{}: {}".format(val_key, len(values[val_key])))

    # ks_result = kstest(values[val_key], norm_pdf.cdf)
    # print(ks_result.statistic, ks_result.pvalue)

    kde = get_kde(values[val_key])
    plt.plot(pdf_x, kde.evaluate(pdf_x), color='orange')

    val_key = 'incorrect pred'
    plt.hist(values[val_key], bins=n_bins, alpha=0.3, density=True, color='blue', label= val_key)
    pdf_x = get_val_space(values[val_key])
    # norm_pdf = get_norm_pdf(values[val_key])
    # plt.plot(pdf_x, norm_pdf.pdf(pdf_x), color='blue')
    # skewness = skew(values[val_key])
    # print('skewness:' , skewness)
    # plt.plot(pdf_x, skewnorm.pdf(pdf_x, skewness), color='blue')

    # # incor_area = 1 - norm_pdf.cdf([0])
    # plt.plot(pdf_x, ecdf(values[val_key]), label=val_key, color='purple')
    incor_area = 1 - ecdf(values[val_key], 0)
    print("{}: {}".format(val_key, len(values[val_key])))

    # ks_result = kstest(values[val_key], norm_pdf.cdf)
    # print(ks_result.statistic, ks_result.pvalue)

    kde = get_kde(values[val_key])
    plt.plot(pdf_x, kde.evaluate(pdf_x), color='blue')

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

def get_dv_dstr(model: Model.prototype, dataloader, clf:Detector.SVM):
    dataset_gts, dataset_preds, _ = model.eval(dataloader)
    dv, _ = clf.predict(dataloader, compute_metrics=False)
    cor_mask = (dataset_gts == dataset_preds)
    incor_mask = ~cor_mask
    cor_dv = dv[cor_mask]
    incor_dv = dv[incor_mask]
    return cor_dv, incor_dv
    
def run(clf:Detector.SVM, dataloader, model_config: Config.OldModel, removal_ratio=0, n_bins = 20):
    cor_dv, incor_dv = get_dv_dstr(model_config, dataloader, clf)
    total_dv = {
        'correct pred': cor_dv,
        'incorrect pred': incor_dv
    }
    fig_path = get_fig_name(model_config.model_dir, model_config.base_type, model_config.model_cnt, removal_ratio)

    # clf_metrics = {
    #     'SVM ': np.round(precision, decimals=2),
    #     'Model': np.round(cor_mask.mean()*100, decimals=2)
    # }
    intersection_area = overlap_plot(total_dv, n_bins=n_bins, path=fig_path, clf_metrics=None, removal_ratio = removal_ratio)
    return intersection_area

def base_plot(value, label, color, pdf_method=None, range=None):
    plt.hist(value, bins= 10 , alpha=0.3, density=True, color=color, label=label, range=range)
    if pdf_method != None:
        dstr = get_pdf(value, pdf_method)
        pdf_x = get_val_space(value)
        if pdf_method == 'norm':
            plt.plot(pdf_x, dstr.pdf(pdf_x), color=color)
        else:
            plt.plot(pdf_x, dstr.evaluate(pdf_x), color=color)

def get_pdf(value, method):
    if method == 'norm':
        return get_norm_pdf(value)
    else:
        return get_kde(value)