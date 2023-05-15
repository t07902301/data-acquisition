from utils.strategy import *
from utils.set_up import set_up
import random
batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, dataset_splits_list, device_config = set_up(1, False)
acc_list, score_list = [], []
old_model_config = Config.OldModel(batch_size,superclass_num, '', device_config, 0)
dataset_splits = dataset_splits_list[0]
dataset_splits = Dataset.DataSplits(dataset_splits, old_model_config.batch_size) # Get dataset splits (train, val, test)

split_data = dataset_splits.dataset
for spit_name in split_data.keys():
    print(spit_name, len(split_data[spit_name]))

clip_processor = CLF.CLIPProcessor(ds_mean=config['data']['mean'], ds_std=config['data']['std']) 
svm = CLF.SVM(dataset_splits.loader['val'])
test_embedding, test_labels, test_idx = clip_processor.evaluate_clip_images(dataset_splits.loader['test'])

# split_names = ['total', 'train', 'val']
split_names = ['total']
for split in split_names:
    # subset = torch.utils.data.Subset(dataset_splits.dataset[split], np.arange(5000))
    subset = dataset_splits.dataset[split]
    batch_size = 500
    loader = torch.utils.data.DataLoader(subset, batch_size= batch_size, num_workers=config['num_workers'], shuffle=False)
    embedding_0, labels_0, idx_0 = clip_processor.evaluate_clip_images(loader)
    sorting = np.argsort(idx_0)
    sorted_emb= embedding_0[sorting]
    sorted_labels = labels_0[sorting]
    shuffle_idx = np.arange(len(subset))
    random.shuffle(shuffle_idx)
    shuffle_emb = embedding_0[shuffle_idx]
    shuffle_labels = labels_0[shuffle_idx]
    # batch_size = 64
    # loader = torch.utils.data.DataLoader(subset, batch_size= batch_size, num_workers=config['num_workers'], shuffle=True)
    # embedding_1, labels_1 = clip_processor.evaluate_clip_images(loader)
    

    score_2 = svm.fitter.base_fit(labels_0, embedding_0)
    acc_2 = svm.fitter.base_predict(test_labels, test_embedding)
    score_1 = svm.fitter.base_fit(sorted_labels, sorted_emb)
    acc_1 = svm.fitter.base_predict(test_labels, test_embedding) 
    score_0 = svm.fitter.base_fit(shuffle_labels, shuffle_emb)
    acc_0 = svm.fitter.base_predict(test_labels, test_embedding) 
    # score_1 = svm.fitter.base_fit(labels_1, embedding_1)
    # acc_1 = svm.fitter.base_predict(test_labels, test_embedding)
    print(score_2, acc_2, score_1, acc_1, score_0, acc_0)