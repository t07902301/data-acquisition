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

def get_correctness_dstr(model, detector, dataloader, pdf_type, correctness:bool):
    '''
    Get decision value distribution of a dataloader against a base model
    '''
    target_dv =  DataStat.get_correctness_dv(model, dataloader, detector, correctness=correctness)
    dataloader_size = DataStat.get_dataloader_size(dataloader)
    prior = (len(target_dv)) / dataloader_size
    dstr = disrtibution(prior, get_pdf(target_dv, pdf_type))
    return dstr

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