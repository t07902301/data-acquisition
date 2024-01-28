import numpy as np
import utils.objects.model as Model
import utils.objects.dataloader as dataloader_utils
from utils.logging import *
from sklearn.mixture import GaussianMixture
from abc import abstractmethod

class Base():
    '''
    Return the ground truth label for weakness detectors
    '''
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def run(self, data_loader, model:Model.Prototype, loader_gts):
        pass

import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def plot_norm_dstr(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma)) 

class Entropy(Base):
    def __init__(self) -> None:
        super().__init__()

    def get_entropy(self, probabs):
        entropy = -np.sum(probabs * np.log2(probabs), axis=1)
        return entropy.reshape((len(entropy), 1))

    def fit(self, base_model:Model.Prototype, data_loader):
        _, _, probab = base_model.eval(data_loader)
        loss = self.get_entropy(probab)
        
        # loss = base_model.ce(data_loader)
        self.model = GaussianMixture(n_components=2, random_state=0).fit(loss)
        self.low_entropy_cluster = np.argmin(self.model.means_)

        mu_0, mu_1 = self.model.means_[0], self.model.means_[1]
        var_0, var_1 = self.model.covariances_[0], self.model.covariances_[1]
        sigma_0, sigma_1 = math.sqrt(var_0[0][0]), math.sqrt(var_1[0][0])
        plot_norm_dstr(mu_0, sigma_0)
        plot_norm_dstr(mu_1, sigma_1)
        plt.savefig('figure/entropy_dstr.png')
        plt.close()
  
    def run(self, data_loader, model:Model.Prototype, loader_gts):
        gts, _, probab = model.eval(data_loader)
        assert (gts != loader_gts).sum() == 0, 'Train Loader Shuffles!: {}'.format((gts != loader_gts).sum())       
        entropy = self.get_entropy(probab)
        low_entropy_mask = (self.model.predict(entropy) == self.low_entropy_cluster)
        data_loader_size = dataloader_utils.get_size(data_loader)
        low_entropy = np.zeros(data_loader_size, dtype = int)
        low_entropy[low_entropy_mask] = 1
        return low_entropy
    
    # def run(self, data_loader, base_model:Model.Prototype=None, metrics=None):
    #     gts, pred, probab = base_model.eval(data_loader)
    #     entropy = self.get_entropy(probab)
    #     weakness_score = self.model.predict_proba(entropy)[self.high_entropy]
    #     if metrics!=None:
    #         gmm_pred = self.model.predict(entropy)
    #         gmm_pred_weakness_mask = (gmm_pred == self.high_entropy)
    #         model_weakness_mask = (gts != pred)
    #         metrics = (gmm_pred_weakness_mask == model_weakness_mask).mean() * 100
    #     else: 
    #         metrics = None

    #     return weakness_score, metrics


class Correctness():
    def __init__(self) -> None:
        super().__init__()

    def run(self, data_loader, model:Model.Prototype, loader_gts):
        '''
        Base Model Prediction Correctness as True Labels for Detectors
        ''' 
        gts, preds, _ = model.eval(data_loader)
        assert (gts != loader_gts).sum() == 0, 'Train Loader Shuffles!: {}'.format((gts != loader_gts).sum())
        correctness_mask = (gts == preds)
        logger.info('Model Acc in Detector Traning Data: {}'.format(correctness_mask.mean()))
        data_loader_size = dataloader_utils.get_size(data_loader)
        correctness = np.zeros(data_loader_size, dtype = int)
        correctness[correctness_mask] = 1
        return correctness
    
def factory(option, model:Model.Prototype, data_loader):
    if option == 'entropy':
        weakness_loader = Entropy()
        weakness_loader.fit(model, data_loader)
        return weakness_loader
    elif option == 'correctness':
        return Correctness()
    else:
        logger.info('Weakness is not Defined.')
        exit()
