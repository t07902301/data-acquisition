from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle as pkl
import os
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import curve_fit
from typing import List
import sys
sys.path.append('/home/yiwei/data-acquisition/')
from utils.logging import *
from sklearn.metrics import r2_score

def power_law(x, a, b):
    return (b*x**(-a))

class PL():
    def __init__(self) -> None:
        pass
    def fit(self, X, y):
        popt, _ = curve_fit(power_law, X, y)
        self.model = {'a': -popt[0], 'b':popt[1]}
    def score(self, X, y):
        pred = power_law(X, self.model['a'], self.model['b'])
        return r2_score(y, pred)
    def predict(self, X):
        return power_law(X, self.model['a'], self.model['b'])

class regressor():
    def __init__(self, dev) -> None:
        self.dev_mode = dev
        self.scaler = None
        if dev == 'rs':
            # self.model = LinearRegression()
            self.model = Ridge()
            # self.model = SVR(kernel='linear')
            # self.model = PL()
            # self.model = SVR(kernel='rbf')

        else: 
            self.model = SVR(kernel='rbf')

        # kernel = RBF(length_scale_bounds = (0.1, 1.0))
        # self.model = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
        # self.model = MLPRegressor(random_state=1, hidden_layer_sizes=(16, 8), learning_rate='adaptive', max_iter=100)

    def scale_inputs(self, inputs):
        if self.scaler == None:
            self.scaler = StandardScaler()
            self.scaler.fit(inputs)
        inputs = self.scaler.transform(inputs)
        return inputs

    def process_inputs(self, inputs):

        if self.dev_mode == 'dv':
            X = [[each[0], each[2]] for each in inputs]
            X = self.scale_inputs(X)

        else:
            X = [each[0] for each in inputs]
            X = np.array(X).reshape(-1, 1)

        y = [each[1] for each in inputs]
        return X, y
    
    def train(self, inputs):

        X, y = self.process_inputs(inputs)

        self.model.fit(X, y)

    def test(self, inputs):

        X, y = self.process_inputs(inputs)

        return self.model.score(X, y)
    
    def predict(self, inputs):
        return self.model.predict(inputs)

def split_train_test(pairs):
    np.random.seed(0)
    pair_size = len(pairs)
    pair_indices = np.arange(pair_size)
    train_indices_mask = np.random.choice(pair_indices, int(pair_size * 0.8))
    train_indices = pair_indices[train_indices_mask]
    test_indices  = pair_indices[~train_indices_mask]
    train = [pairs[i] for i in train_indices]
    test = [pairs[i] for i in test_indices]
    return train, test
    
def import_file(model_dir, dev):
   
    file = os.path.join('log/{}/reg/{}.pkl'.format(model_dir, dev))
   
    with open(file, 'rb') as f:
        out = pkl.load(f)

    logger.info('load from {}'.format(file))

    return out, file

import matplotlib.pyplot as plt

def plot(inputs, label):
    inputs = sorted(inputs, key=lambda tup: tup[0])
    # inputs.sort(key=lambda tup: tup[0])
    X = [each[0] for each in inputs]
    y = [each[1] for each in inputs]
    plt.plot(X, y, label=label)

def predict(epochs, models: List[regressor], n_samples, probabs=None):
    
    test_inputs = []

    if probabs != None:
        for epo in range(epochs):
            test_inputs_epoch = []
            for idx, n_sample in enumerate(n_samples): 
                test_inputs_epoch.append([probabs[epo][idx], n_sample])
            test_inputs.append(models[epo].scaler.transform(test_inputs_epoch))
    else:
        n_samples =  np.array(n_samples).reshape(-1, 1)
        test_inputs = [n_samples for i in range(epochs)]

    pred_list = []

    for epo in range(epochs):
        model = models[epo]
        pred = model.predict(test_inputs[epo])
        pred_list.append(pred)
    
    logger.info('avg pred: {}'.format(np.round(np.mean(pred_list, axis=0), decimals=3)))
    logger.info('all preds: {}'.format(pred_list))

def train_test(epochs, pairs, dev) -> List[regressor]:

    score_list, model_list = [], []

    for epo in range(epochs):

        train_data, test_data = split_train_test(pairs[epo])
        model = regressor(dev)
        model.train(train_data)
        score = model.test(test_data)
        score_list.append(score)
        model_list.append(model)

    logger.info('avg score: {}'.format(np.round(np.mean(score_list), decimals=3)))
    logger.info('scores: {}'.format(score_list))

    return model_list

def main(epochs, model_dir, dev):

    fh = logging.FileHandler('log/{}/reg_{}.log'.format(model_dir, dev), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    pairs, file = import_file(model_dir, dev)

    plot_regress(epochs, pairs, file)

    models = train_test(epochs, pairs, dev)

    # # probab_list = [[0.127151728145905, 0.15725853873628695, 0.1890850242204149, 0.21795223942229652, 0.24617770115737173], [0.6924500115603807, 0.872724325949984, 0.9377391251474637, 0.9377280516334007, 0.9377280516334007], [0.8268993571274925, 0.9254674029451055, 0.9254674029451055, 0.9456753815953349, 0.9454728169369965], [0.7336369264799719, 0.7336369264799719, 0.8494535193131926, 0.8992251491901697, 0.8992251491901697], [0.7845818641998934, 0.7845818641998934, 0.7845818641998934, 0.7871409909645718, 0.9446297865104551]]
    n_samples = [125, 225, 325, 425, 525, 625]

    predict(epochs, models, n_samples)

def plot_gt(epochs, model_dir, dev):

    fh = logging.FileHandler('log/{}/gt_{}.log'.format(model_dir, dev), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    x = [[0.13192432664716852, 0.13977537888673375, 0.14369696231658657, 0.150766216514042, 0.16324239211597927, 0.17920451209593616], [0.11231019761475364, 0.1453124084323593, 0.16972437829499215, 0.19301595436027608, 0.22065587278659257, 0.24599441787737222], [0.10633929442819588, 0.10785733202902527, 0.11036457845547994, 0.11666931881002483, 0.1283583717369866, 0.14204542674354836], [0.08300015582166108, 0.12835756275746812, 0.16601244565976622, 0.19447076692733825, 0.2166034373666523, 0.2393332084151524], [0.07418545503395561, 0.11666503674641472, 0.14373744810235398, 0.1679257321020891, 0.19072406850477913, 0.21893561536917447], [0.09531377469105817, 0.13064324313451048, 0.15850928574386802, 0.1823575182339508, 0.20584213633658668, 0.2303439740556444], [0.07905119545020327, 0.11761782220930801, 0.14096009201977766, 0.16019047063989889, 0.1854538785270153, 0.2205152246473749], [0.10599917293009407, 0.13673843537616154, 0.15603320549386188, 0.1720428983369244, 0.18819087109256163, 0.20713118558851756], [0.07686832750477242, 0.1087641506906076, 0.1436913107837526, 0.1707414391652311, 0.20588399541395075, 0.24151562377803118], [0.09766813098362864, 0.12118522841953706, 0.1327170151660317, 0.14626748814686377, 0.16090540599821385, 0.1791159225616521]]
    y = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 3.0, 4.0, 3.0, 3.0], [5.0, 6.0, 6.0, 6.0, 6.0, 6.0], [1.0, 1.0, 2.0, 2.0, 2.0, 2.0], [5.0, 2.0, 4.0, 4.0, 1.0, 4.0], [3.0, 4.0, 3.0, 3.0, 3.0, 3.0], [3.0, 4.0, 4.0, 4.0, 4.0, 2.0], [5.0, 5.0, 4.0, 4.0, 4.0, 4.0]]

    # for epo in range(epochs):
    #     if epo == 0 or epo == 2:
    #         continue   
    #     pair_epo = []
    #     for idx in range(len(x[epo])):
    #         pair_epo.append((x[epo][idx], y[epo][idx]))
    #     plot(pair_epo, str(epo))

    epochs = [4]
    for epo in epochs:
        pair_epo = []
        for idx in range(len(x[epo])):
            pair_epo.append((x[epo][idx], y[epo][idx]))
        plot(pair_epo, str(epo))
        
    plt.legend()
    plt.show()
    file = 'log/{}/gt_{}.png'.format(model_dir, dev)
    plt.savefig(file)
    logger.info('figure save to {}'.format(file))

def plot_regress(epochs, pairs, file:str):
   
    # for epo in range(epochs):

    #     if epo == 0 or epo == 2:
    #         continue

    #     plot(pairs[epo], str(epo))
    
    epochs = [4]
    for epo in epochs:
        plot(pairs[epo], str(epo))

    file = file.replace('.pkl', '.png')
    plt.legend()
    plt.show()
    plt.savefig(file)
    logger.info('figure save to {}'.format(file))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    args = parser.parse_args()
    main(args.epochs, args.model_dir, args.dev)
    # plot_gt(args.epochs, args.model_dir, args.dev)
