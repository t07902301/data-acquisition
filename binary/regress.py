from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
import os
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# from sklearn.neural_network import MLPRegressor
from typing import List
class regressor():
    def __init__(self) -> None:
        # self.model = LinearRegression()
        self.model = Ridge()
        # self.model = SVR(kernel='rbf')

        # kernel = RBF(length_scale_bounds = (0.1, 1.0))
        # self.model = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
        # self.model = MLPRegressor(random_state=1, hidden_layer_sizes=(16, 8), learning_rate='adaptive', max_iter=100)

    def train(self, inputs):
        X = [each[0] for each in inputs]
        y = [each[1] for each in inputs]
        X = np.array(X).reshape(-1, 1)
        scaler =None

        # scaler = MinMaxScaler()
        # scaler.fit(X)
        # X = scaler.transform(X)
        # y = np.array(y).reshape(-1, 1)
        # y = scaler.transform(y)

        self.model.fit(X, y)

        # y_pred, y_pred_std = self.model.predict(X, return_std=True)

        # plt.plot(X, y, 'ko', label = 'Training Data')
        # plt.plot(x1, y1, 'b-', label = "Predicted Function Mean")
        # plt.title("Zero Shear Viscosity of Wormlike Micelles")
        # plt.xlabel('X')
        # plt.ylabel('y')

        # # Plotting the uncertainty
        # y1 = y1.flatten()
        # plt.fill_between(x1, y1 - y1std, y1 + y1std, alpha=0.3, color='k', label="Uncertainty")

        # plt.xlabel("log (Salt concentration)")
        # plt.ylabel("log (zero shear viscosity)")
        # plt.legend()

        return scaler

    def test(self, inputs, scaler: MinMaxScaler):
        X = [each[0] for each in inputs]
        y = [each[1] for each in inputs]
        X = np.array(X).reshape(-1, 1)

        # X = scaler.transform(X)
        # y = np.array(y).reshape(-1, 1)
        # y = scaler.transform(y)

        return self.model.score(X, y)
    
    def predict(self, inputs):
        inputs = np.array(inputs).reshape(-1, 1)
        return self.model.predict(inputs)

def regression_split_train_test(pairs):
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
   
    file = os.path.join('log/{}/dev'.format(model_dir), 'val_{}.pkl'.format(dev))
   
    with open(file, 'rb') as f:
        out = pkl.load(f)

    print('load from', file)

    return out

import matplotlib.pyplot as plt

def plot(inputs):
    inputs.sort(key=lambda tup: tup[0])
    X = [each[0] for each in inputs]
    y = [each[1] for each in inputs]
    plt.plot(X, y)
    plt.show()

def plot_regress(epochs, pairs, model_dir, dev_mode):

    for epo in range(epochs):

        plot(pairs[epo])
    
    file = os.path.join('log/{}/dev'.format(model_dir), '{}.png'.format(dev_mode))

    plt.savefig(file)

    print('figure save to', file)

def predict(epochs, pairs, models: List[regressor], test_list):

    pred_list = []

    for epo in range(epochs):

        model = models[epo]

        pred = model.predict(test_list)
        pred_list.append(pred)
    
    print(np.round(np.mean(pred_list, axis=0), decimals=3))
    print(*pred_list)

def train(epochs, pairs) -> List[regressor]:

    score_list, model_list = [], []

    for epo in range(epochs):

        train, test = regression_split_train_test(pairs[epo])
        model = regressor()
        scaler = model.train(train)
        score = model.test(test, scaler)
        score_list.append(score)
        model_list.append(model)

    print(np.round(np.mean(score_list), decimals=3))
    print(score_list)

    return model_list

def main(epochs, model_dir, dev):

    pairs = import_file(model_dir, dev)

    test_list = [225, 275, 325, 375, 425]
    # test_list = [225, 325, 425, 525, 625]

    # plot_regress(epochs, pairs, model_dir, dev)

    models = train(epochs, pairs)

    predict(epochs, pairs, models, test_list)


import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-dev','--dev',type=str, default='dv')

    args = parser.parse_args()
    main(args.epochs, args.model_dir, args.dev)
