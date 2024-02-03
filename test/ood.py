import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))
from sklearn.svm import OneClassSVM
from utils.set_up import *
import numpy as np
import utils.objects.dataloader as dataloader_utils
import utils.objects.Detector as Detector
import utils.dataset.wrappers as dataset_utils
from sklearn.linear_model import SGDOneClassSVM
import sklearn.metrics as sklearn_metrics
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from novelty.src.dev import dev
from utils.logging import *

def get_raw_novelty(dataset, known_labels):
    novelty = []
    for idx in range(len(dataset)):
        label = 1 if dataset[idx][2] in known_labels else -1
        novelty.append(label)
    return np.array(novelty)

def get_cover_samples(dataset, is_covered):

    cover_indicies = np.arange(len(dataset))[is_covered == 1]

    return torch.utils.data.Subset(dataset, cover_indicies)

def run(ds:dataset_utils.DataSplits, clip_processor, known_labels):

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

def get_novelty_label(dataset:dataset_utils.cifar.Novelty):
    labels = [dataset[idx][1] for idx in range(len(dataset))]
    return np.array(labels)

def svd(ds:dataset_utils.DataSplits, known_labels, batch_size):

    # train_dataset = dataset_utils.cifar.Novelty(ds.dataset['val_shift'], known_labels)

    train_loader = ds.loader['val_shift']

    test_dataset = dataset_utils.cifar.Novelty(ds.dataset['market'], known_labels)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    svd_indices = dev(train_loader, test_loader, xp_path='log/dev')
    
    cover_ds = dataset_utils.get_indices_by_labels(ds.dataset['market'], known_labels)

    cover_size = len(cover_ds)

    pred_indices =  svd_indices[ : cover_size]

    pred_ds = torch.utils.data.Subset(ds.dataset['market'], pred_indices)

    pred = np.zeros(len(pred_ds)) # normal instances

    gt = get_novelty_label(pred_ds)

    return (sklearn_metrics.balanced_accuracy_score(gt, pred)*100)

def main(epochs,  model_dir ='', device=1):

    fh = logging.FileHandler('log/{}/ood.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    parse_args, ds_list, normalize_stat = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    odd_acc_list = []

    known_labels = parse_args.general_config['data']['labels']['cover']['target']

    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        ds = ds_list[epo]
        ds = dataset_utils.DataSplits(ds, 32)
        # ood_acc = svd( ds, known_labels, batch_size['new'])
        ood_acc = run( ds, clip_processor, known_labels)
        odd_acc_list.append(ood_acc)

    logger.info('ood acc: {}'.format(odd_acc_list))
    logger.info('ood avg acc: {}'.format(np.round(np.mean(odd_acc_list), decimals=3)))

    split_data = ds.dataset
    for spit_name in split_data.keys():
        logger.info('{}:{}'.format(spit_name, len(split_data[spit_name])))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-op','--option',type=str, default='object')
    parser.add_argument('-ds','--dataset',type=str, default='core')

    args = parser.parse_args()
    main(args.epochs,args.model_dir)