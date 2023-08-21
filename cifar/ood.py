from sklearn.svm import OneClassSVM
from utils.set_up import *
import numpy as np
import utils.objects.dataloader as dataloader_utils
import utils.objects.Detector as Detector
from sklearn.linear_model import SGDOneClassSVM
import sklearn.metrics as sklearn_metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from novelty.src.dev import dev

def get_raw_novelty(dataset, known_labels):
    novelty = []
    for idx in range(len(dataset)):
        label = 1 if dataset[idx][2] in known_labels else -1
        novelty.append(label)
    return np.array(novelty)

def get_cover_samples(dataset, is_covered):

    cover_indicies = np.arange(len(dataset))[is_covered == 1]

    return torch.utils.data.Subset(dataset, cover_indicies)

def run(ds:Dataset.DataSplits, clip_processor, known_labels):

    ref_latent, _ = dataloader_utils.get_latent(ds.loader['val_shift'], clip_processor, 'clip')

    clf = OneClassSVM(gamma='auto', kernel='rbf').fit(ref_latent)
    # clf = SGDOneClassSVM(random_state=0).fit(ref_latent)
    # clf = IsolationForest(random_state=0).fit(ref_latent)
    # clf = LocalOutlierFactor(novelty=True).fit(ref_latent)

    test_latent, _ = dataloader_utils.get_latent(ds.loader['market'], clip_processor, 'clip')

    pred_novelty = clf.predict(test_latent)

    gt_novelty = get_raw_novelty(ds.dataset['market'], known_labels)

    score = sklearn_metrics.precision_score(gt_novelty, pred_novelty, pos_label=1) * 100

    return score

def get_novelty_label(dataset:Dataset.cifar.Novelty):
    labels = [dataset[idx][1] for idx in range(len(dataset))]
    return np.array(labels)

def svd(ds:Dataset.DataSplits, known_labels, batch_size):

    # train_dataset = Dataset.cifar.Novelty(ds.dataset['val_shift'], known_labels)

    train_loader = ds.loader['val_shift']

    test_dataset = Dataset.cifar.Novelty(ds.dataset['market'], known_labels)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    svd_indices = dev(train_loader, test_loader, xp_path='log/dev')
    
    known_indices = Dataset.get_indices_by_labels(ds.dataset['market'], known_labels)

    known_size = len(known_indices)

    pred_indices =  svd_indices[:known_size]

    pred_ds = torch.utils.data.Subset(ds.dataset['market'], pred_indices)

    pred = np.zeros(len(pred_ds)) # normal instances

    gt = get_novelty_label(pred_ds)

    return (sklearn_metrics.balanced_accuracy_score(gt, pred)*100)

def main(epochs,  model_dir =''):
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, 0)
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    odd_acc_list = []

    known_labels = config['data']['cover_labels']['target']

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        ds = Dataset.DataSplits(ds_list[epo], 32)

        # ood_acc = svd( ds, known_labels, 32)
        ood_acc = run( ds, clip_processor, known_labels)

        odd_acc_list.append(ood_acc)

    print(odd_acc_list)
    print(np.round(np.mean(odd_acc_list), decimals=3))

    split_data = ds.dataset
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.epochs,args.model_dir)