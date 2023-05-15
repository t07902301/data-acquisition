from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter

def main(epochs, new_model_setter='retrain', model_dir ='', device=0):
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    results = []

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, True, new_model_setter, False)
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)
        base_model = Model.load(old_model_config)
        clf = CLF.SVM(dataset_splits.loader['train_clip'])
        score = clf.fit(base_model, dataset_splits.loader['val_shift'])
        dv = {}
        for split_name, loader in dataset_splits.loader.items():
            dv[split_name],_ =clf.predict(loader)
        results.append(dv)

    result_plotter = Plotter.Histogram(new_model_config)
    result_plotter.run(epochs, results)  

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device)