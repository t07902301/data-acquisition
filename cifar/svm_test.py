from utils.strategy import *
from utils.set_up import set_up

old_labels = set(Dataset.data_config['train_label']) - set(Dataset.data_config['remove_fine_labels'])
def remove_old_data(dataset, sample_ratio):
    _, left_data = Dataset.split_dataset(dataset, old_labels, sample_ratio)
    return left_data
def main(epochs,  model_dir ='', device_id=0, base_type=''):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    clip_processor = Detector.load_clip(device_config)

    # old_data_ratio_list, _ = np.linspace(0, 1, retstep=True, num=5)
    old_data_ratio_list = [0, 0.75]
    old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, 0, base_type)
    dataset = ds_list[0]

    type_1 = [3, 42, 43, 88, 63, 64, 66, 75]
    type_2 =  [97, 34]
    ts = dataset['market']

    ts_1_1, _ = Dataset.split_dataset(ts, type_1, 0.25)
    ts_1_2, _ = Dataset.split_dataset(ts, type_2, 0.5)

    ts_2_1, _ = Dataset.split_dataset(ts, type_1, 0.25)
    ts_2_2, _ = Dataset.split_dataset(ts, type_2, 0.5)

    ts_1 = torch.utils.data.ConcatDataset([ts_1_1, ts_1_2])
    ts_2 = torch.utils.data.ConcatDataset([ts_2_1, ts_2_2])
    ts_1_loader = torch.utils.data.DataLoader(ts_1, batch_size=batch_size['base']) 
    ts_2_loader = torch.utils.data.DataLoader(ts_2, batch_size=batch_size['base'])    

    # print(len(ts_1_1) / len(ts_1), len(ts_1_2) / len(ts_1))
    # print(len(ts_2_1) / len(ts_2), len(ts_2_2) / len(ts_2))

    # ts_flat = Detector.get_flattened(ts_1_loader)
    # print(ts_flat.shape)

    # vs = dataset['val_shift']
    # ms = dataset['market']
    # vs_loader = torch.utils.data.DataLoader(vs, batch_size=batch_size['base']) 
    # ms_loader = torch.utils.data.DataLoader(ms, batch_size=batch_size['base'])

    # vs_1 = Dataset.get_subset_by_labels(vs, type_1)
    # vs_2 = Dataset.get_subset_by_labels(vs, type_2)
    # print(len(vs_1) / len(vs), len(vs_2) / len(vs))

    # ms_1 = Dataset.get_subset_by_labels(ms, type_1)
    # ms_2 = Dataset.get_subset_by_labels(ms, type_2)
    # print(len(ms_1) / len(ms), len(ms_2) / len(ms))

    base_model = Model.resnet(2)
    base_model.load(old_model_config)

    # gt,pred,_  = base_model.eval(ts_1_loader)
    # acc = (gt==pred).mean()*100
    # print(acc)
    # gt,pred,_  = base_model.eval(ts_2_loader)
    # acc = (gt==pred).mean()*100
    # print(acc)

    # clf = Detector.SVM(ts_1_loader, clip_processor, split_and_search=True)
    # for C_ in np.logspace(-6, 0, 7, endpoint=True):
    #     score = clf.fit(base_model, ts_1_loader, C_)
    #     clf.predict(ts_1_loader, compute_metrics=True, base_model=base_model)
        # clf.predict(ts_2_loader, compute_metrics=True, base_model=base_model)

    # clf = Detector.SVM(ts_1_loader, clip_processor, split_and_search=True)
    # score = clf.fit(base_model, ts_1_loader)
    # clf = Detector.SVM(ts_1_loader, clip_processor, split_and_search=False)
    # score = clf.fit(base_model, ts_1_loader)
    clf = Detector.SVM(ts_1_loader, clip_processor, split_and_search=True)
    score = clf.fit(base_model, ts_1_loader)
    clf.predict(ts_1_loader, compute_metrics=True, base_model=base_model)
    clf.predict(ts_2_loader, compute_metrics=True, base_model=base_model)

    

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='resnet')

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir, args.device, args.base_type)