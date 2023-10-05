# Data Acquisition for Generalizing Black-Box Models

## Set up

Under the currect directory, three folder need to be set to keep outputs from experiments.

1. **log**: keep all output from logger.
    - config.yaml: model training, detector training, data split and shift. **config.yaml** under *config* folder is a template. Configs for all tasks are also in *config* folder.

2. **model**: keep new models, acquired data, updated detectors.   

3. available data: **data** directory contains indices of designed data shifts from raw datasets 

   - Raw datasets: 

    1. [Cifar-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz): Indicate the path to the downloaded zip in the "root" field of the config file

    2. [Core-50](http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz): the original "core50_imgs.npz" file has images compressed to 32x32 and sampled off 30 frames from each object and session. The new dataset is called "core.pkl". 

model_directory with the format of - (dataset name) _ task _ (other info) - is used to name all relevant outputs in **log**, **model**, and **data**. 

Examples of model_directory: core_object_resnet, cifar-4class

## Set up Source Models

**pretrain.py**: train source models and evaluate them before and after data shifts.

## Data Acquisition + New Model Generation

**utils/strategy.py**: acquisition strategy + workspace (source model, data splits, detector if needed.) + build and save new models

**utils/ood.py**: filter out irrelevant data from the data pool. build novelty detector (one-class SVM) in the validation set, and return detection accuracy.   

- If the precision is bigger than threshold,the proportion of target labels in filtered market, then market filtering should work.

**main.py** : run the acquisition. 

## Test New Model

**utils/checker.py**: ensemble methods (AE, WTA, partition by feature scores)

**test.py**: evaluate models from acquisition

## Acquisition Statistics

**pretrain.py**: base model performance before and after data shifts; classification accuracy on c_w and c_\not{w} in detector

**seq_stat.py**: only run sequential strategy and return statistics of detection accuracy or misclassification in acquired data. No model building and saving. 

**stat_check.py**: the distribution of acquired data; test data under WTA or partition; final detector from sequential acquisition. 

## Dataset Generation

Shifted data splits are generated first by split raw dataset into 4 splits (train, test, validation and data pool), and then make data shifts by removing some labels from train split. By far, we first save the indices of 4 data splits and statistics for data normalization into **init_data** directory. Next we save data shifts indices into **data** directoty. 

**data_setup.py**: use "save mode" parameter to choose which indices to save (split or shift). 

Preliminary processing of Core 50 can be found in *Basic Process*
1. **core.ipynb**: 'core50_imgs.npz' -> resize to 32x32 and transform labels -> 'core_data.pkl'
    - An [auxlirary file](https://vlomonaco.github.io/core50/data/paths.pkl) is used to extract labels from "core50_imgs.npz"

2. **meta.py**: sample frames from indicated categories from Core-50. ('core_data.pkl' -> sample frames -> 'core.pkl')

