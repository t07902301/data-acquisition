from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import pickle as pkl
import os

class linear_regressor():
    def __init__(self) -> None:
        # self.model = LinearRegression()
        self.model = Ridge()
    
    def train(self, inputs):
        X = [each[0] for each in inputs]
        y = [each[1] for each in inputs]
        X = np.array(X).reshape(-1, 1)
        self.model.fit(X, y)

    def test(self, inputs):
        X = [each[0] for each in inputs]
        y = [each[1] for each in inputs]
        X = np.array(X).reshape(-1, 1)
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
   
    file = os.path.join('log/{}'.format(model_dir), 'val_{}.pkl'.format(dev))
   
    with open(file, 'rb') as f:
        out = pkl.load(f)

    print('load from', file)

    return out

import matplotlib.pyplot as plt

def plot(inputs):
    inputs.sort(key=lambda tup: tup[0])
    X = [each[0] for each in inputs]
    y = [each[1] for each in inputs]
    
    # fig, ax = plt.subplots()
    plt.plot(X, y)

    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #     title='About as simple as it gets, folks')
    # ax.grid()

    # fig.savefig("test.png")
    plt.show()

def main(epochs, model_dir, dev):

    pairs = import_file(model_dir, dev)

    score_list = []

    test_list = [100, 150, 200, 250, 300, 400, 500]

    pred_list = []

    for epo in range(epochs):

        plot(pairs[epo])

    plt.savefig("test.png")

    #     train, test = regression_split_train_test(pairs[epo])
    #     solver = linear_regressor()
    #     solver.train(train)

    #     # score = solver.test(test)
    #     # score_list.append(score)
        
    #     pred = solver.predict(test_list)
    #     pred_list.append(pred)
  
    # print(np.round(np.mean(pred_list, axis=0), decimals=3))
    # print(pred_list)
    # # print(np.round(np.mean(score_list), decimals=3))
    # # print(score_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-dev','--dev',type=str, default='dv')

    args = parser.parse_args()
    main(args.epochs, args.model_dir, args.dev)
