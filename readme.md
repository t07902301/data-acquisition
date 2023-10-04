# Data Acquisition for Generalizing Models behind APIs under Harmful Data Shifts

## Directory of Results and Data

log/, data/, model/ 

### Configuration Example

## Set up Dataset

1. meta.py: sample frames from indicated categories from Core-50.

2. data_setup.py: indices to data splits from the raw dataset (core-50 with sampled frames.)
    - select indicated classes from Cifar-100.
    - data shift generated.

check out-of-distribution detection precision with ood.py. 
If the precision is bigger than threshold,the proportion of target labels in filtered market, then market filtering should work.

## Set up Base Models

pretrain.py: use data split indices to load actual data from the raw dataset. (sampled core-50 from meta.py)

## Data Acquisition + New Model Generation

strategy.py: acquisition strategy + workspace (base model, data splits, detector if needed.) + build and save new models

ood.py: filter out irrelevant data from the data pool. build novelty detector (one-class SVM) in the validation set, and return detection accuracy.   

main.py: run the acquisition. 


## Test New Model

checker.py: ensemble methods (AE, WTA, partition by feature scores)

test.py: evaluate models from acquisition

## Statistics before, during and after Acquisition

pretrain.py: base model performance before and after data shifts; classification accuracy on c_w and c_\not{w} in detector

seq_stat.py: only run sequential strategy and return statistics of detection accuracy or misclassification in acquired data. No model building and saving. 

stat_check.py: the distribution of acquired data; test data under WTA or partition; final detector from sequential acquisition. 
