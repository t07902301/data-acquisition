from sklearn.svm import OneClassSVM
import numpy as np
import torch
import utils.objects.dataloader as dataloader_utils
import utils.dataset.wrappers as Dataset
# import utils.objects.Detector as Detector
# from sklearn.linear_model import SGDOneClassSVM
import sklearn.metrics as sklearn_metrics
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor
# from novelty.src.dev import dev

def get_raw_novelty(dataset, known_labels):
    novelty = []
    for idx in range(len(dataset)):
        label = 1 if dataset[idx][2] in known_labels else -1
        novelty.append(label)
    return np.array(novelty)

def get_cover_samples(dataset, is_covered):

    cover_indicies = np.arange(len(dataset))[is_covered == 1]

    return torch.utils.data.Subset(dataset, cover_indicies)

def run(ds:Dataset.DataSplits, clip_processor, known_labels, normal_ds = 'val_shift', check_ds = 'market'):

    ref_latent, _ = dataloader_utils.get_latent(ds.loader[normal_ds], clip_processor, 'clip')

    clf = OneClassSVM(gamma='auto', kernel='rbf').fit(ref_latent)
    # clf = SGDOneClassSVM(random_state=0).fit(ref_latent)
    # clf = IsolationForest(random_state=0).fit(ref_latent)
    # clf = LocalOutlierFactor(novelty=True).fit(ref_latent)

    test_latent, _ = dataloader_utils.get_latent(ds.loader[check_ds], clip_processor, 'clip')

    pred_novelty = clf.predict(test_latent)

    gt_novelty = get_raw_novelty(ds.dataset[check_ds], known_labels)

    detect_prec = sklearn_metrics.precision_score(gt_novelty, pred_novelty, pos_label=1) * 100

    cover_samples = get_cover_samples(ds.dataset[check_ds], pred_novelty)

    print('OOD Precision:', detect_prec)

    return cover_samples