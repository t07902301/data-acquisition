from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle as pkl
import os
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPRegressor
from typing import List

class regressor():
    def __init__(self, dev) -> None:
        self.dev_mode = dev
        self.scaler = None
        if dev == 'rs':
            # self.model = LinearRegression()
            # self.model = Ridge()
            self.model = SVR(kernel='linear')
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
   
    file = os.path.join('log/{}/reg'.format(model_dir), '{}.pkl'.format(dev))
   
    with open(file, 'rb') as f:
        out = pkl.load(f)

    print('load from', file)

    return out, file

import matplotlib.pyplot as plt

def plot(inputs):
    inputs = sorted(inputs, key=lambda tup: tup[0])
    # inputs.sort(key=lambda tup: tup[0])
    X = [each[0] for each in inputs]
    y = [each[1] for each in inputs]
    plt.plot(X, y)
    plt.show()

def plot_regress(epochs, pairs, file:str):

    for epo in range(epochs):

        plot(pairs[epo])
    
    file = file.replace('.pkl', '.png')

    plt.savefig(file)

    print('figure save to', file)

def predict(epochs, models: List[regressor], n_samples, probabs=None):
    
    test_inputs = []

    if probabs != None:
        for epo in range(epochs):
            test_inputs_epoch = []
            for idx, n_sample in enumerate(n_samples): 
                test_inputs_epoch.append([probabs[epo][idx], n_sample])
            test_inputs.append(models[epo].scaler.transform(test_inputs_epoch))
    else:
        test_inputs = [n_samples for i in range(epochs)]
        test_inputs = np.array(test_inputs).reshape(-1, 1)

    pred_list = []

    for epo in range(epochs):
        model = models[epo]
        pred = model.predict(test_inputs[epo])
        pred_list.append(pred)
    
    print(np.round(np.mean(pred_list, axis=0), decimals=3))
    for i in pred_list:
        print(*i)

def train_test(epochs, pairs, dev) -> List[regressor]:

    score_list, model_list = [], []

    for epo in range(epochs):

        train_data, test_data = split_train_test(pairs[epo])
        model = regressor(dev)
        model.train(train_data)
        score = model.test(test_data)
        score_list.append(score)
        model_list.append(model)

    print(np.round(np.mean(score_list), decimals=3))
    print(score_list)

    return model_list

def main(epochs, model_dir, dev):

    pairs, file = import_file(model_dir, dev)

    plot_regress(epochs, pairs, file)

    probab_list = [[0.127151728145905, 0.15725853873628695, 0.1890850242204149, 0.21795223942229652, 0.24617770115737173], [0.6924500115603807, 0.872724325949984, 0.9377391251474637, 0.9377280516334007, 0.9377280516334007], [0.8268993571274925, 0.9254674029451055, 0.9254674029451055, 0.9456753815953349, 0.9454728169369965], [0.7336369264799719, 0.7336369264799719, 0.8494535193131926, 0.8992251491901697, 0.8992251491901697], [0.7845818641998934, 0.7845818641998934, 0.7845818641998934, 0.7871409909645718, 0.9446297865104551]]

    n_samples = [225, 325, 425, 525, 625]

    models = train_test(epochs, pairs, dev)

    predict(epochs, models, n_samples, probab_list)

    # print(pairs)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    args = parser.parse_args()
    main(args.epochs, args.model_dir, args.dev)
