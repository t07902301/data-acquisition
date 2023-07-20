import numpy as np
from scipy.stats import norm, gaussian_kde, kstest, skewnorm, skew
import utils.statistics.data as DataStat
import matplotlib.pyplot as plt

def get_intervals(values):
    max_val = max(values)
    min_val = min(values)
    val_space = np.linspace(min_val, max_val)   
    return val_space

def get_norm_pdf(values):
    mean = np.mean(values)
    std = np.std(values)
    dist = norm(mean, std)
    return dist

def get_kde(values):
    kernel = gaussian_kde(values)
    return kernel


def ecdf(raw_values, cut_value=None):
    raw_values = np.array(raw_values)
    n_raw_vals = len(raw_values)
    if cut_value == None:
        intervals = get_intervals(raw_values)
        result = []
        for edge in intervals:
            result.append(np.sum(raw_values <= edge) / n_raw_vals)
        return result
    else:
        return np.array([np.sum(raw_values <= cut_value) / n_raw_vals])


class disrtibution():
    def __init__(self, prior, dstr) -> None:
        self.prior = prior
        self.dstr = dstr

def get_dv_dstr(model, detector, dataloader, pdf_type):
    '''
    Get decision value distribution of a dataloader against a base model
    '''
    cor_dv, incor_dv = DataStat.get_hard_easy_dv(model, dataloader, detector)
    # total_dv = np.concatenate((incor_dv,cor_dv))
    # print('Negative DV: {}%'.format((total_dv<0).mean()*100))
    correct_prior = (len(cor_dv)) / (len(cor_dv) + len(incor_dv))
    correct_dstr = disrtibution(correct_prior, get_pdf(cor_dv, pdf_type))
    incorrect_dstr =  disrtibution(1 - correct_prior, get_pdf(incor_dv, pdf_type))
    return correct_dstr, incorrect_dstr

def get_pdf(value, method):
    if method == 'norm':
        return get_norm_pdf(value)
    else:
        return get_kde(value)
    
def base_plot(value, label, color, pdf_method=None, range=None, n_bins = 10):
    plt.hist(value, bins= n_bins , alpha=0.3, density=True, color=color, label=label, range=range)
    if pdf_method != None:
        dstr = get_pdf(value, pdf_method)
        pdf_x = get_intervals(value)
        if pdf_method == 'norm':
            plt.plot(pdf_x, dstr.pdf(pdf_x), color=color)
        else:
            plt.plot(pdf_x, dstr.evaluate(pdf_x), color=color)
    plt.legend()