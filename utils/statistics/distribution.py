import numpy as np
from scipy.stats import norm, gaussian_kde, kstest, skewnorm, skew
import utils.statistics.data as DataStat
import matplotlib.pyplot as plt
import utils.objects.dataloader as dataloader_utils
from utils.objects.Detector import Prototype as DetecorPrototye
from utils.objects.model import Prototype as ModelPrototye
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
    if cut_value is None:
        intervals = get_intervals(raw_values)
        result = []
        for edge in intervals:
            result.append(np.sum(raw_values <= edge) / n_raw_vals)
        return result
    else:
        return np.array([np.sum(raw_values <= cut_value) / n_raw_vals])

class PDF():
    def __init__(self, type, train_data) -> None:
        self.type = type
        self.function = get_pdf(train_data, type)

    def evaluate(self, value):
        return self.function.pdf(value)
    
    def accumulate(self, max_val, min_val= None):
        if self.type == 'norm':
            return self.function.cdf(max_val)
        elif self.type == 'kde':
            return self.function.integrate_box_1d(min_val, max_val)

class Disrtibution():
    '''
    Get decision value distribution of a dataloader against a given model
    '''
    def __init__(self, detector:DetecorPrototye, dataloader, pdf_type) -> None:
        target_weakness_score, _ = detector.predict(dataloader)
        self.pdf = PDF(pdf_type, target_weakness_score)
        self.type = pdf_type

class CorrectnessDisrtibution():
    '''
    Prior probability; \n Probability densiity function; \n Correct Predictions or not
    '''
    def __init__(self, model: ModelPrototye, detector:DetecorPrototye, dataloader, pdf_type, correctness: bool) -> None:
        target_weakness_score =  DataStat.get_correctness_weakness_score(model, dataloader, detector, correctness=correctness)
        dataloader_size = dataloader_utils.get_size(dataloader)
        self.prior = len(target_weakness_score) / dataloader_size
        self.pdf = PDF(pdf_type, target_weakness_score)
        self.correctness = correctness

def get_pdf(value, method):
    if method == 'norm':
        return get_norm_pdf(value)
    else:
        return get_kde(value)
    
def base_plot(value, label, color, pdf_method=None, range=None, n_bins = 10, hatch_style = '/', line_style='-', alpha=1):
    plt.hist(value, bins= n_bins , alpha=alpha, density=True, color=color, label=label, range=range, fill=True, hatch=hatch_style, edgecolor='k', histtype='step')
    if pdf_method != None:
        dstr = get_pdf(value, pdf_method)
        pdf_x = get_intervals(value)
        if pdf_method == 'norm':
            plt.plot(pdf_x, dstr.pdf(pdf_x), color='black', linestyle=line_style)
        else:
            plt.plot(pdf_x, dstr.evaluate(pdf_x), color='black', linestyle=line_style)
    plt.legend(fontsize=15)
