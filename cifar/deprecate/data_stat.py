from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.subset as Subset
import utils.statistics.checker as Checker
import utils.statistics.plot as Plotter

def compare(acquisition_config:Config.Acquistion, model_config:Config.NewModel, product:Checker.subset, dataset_splits:Dataset.DataSplits, bound):
    test_dv = product.test_info['dv']
    new_test_mask = test_dv<=bound
    old_test_mask = ~new_test_mask
    dv_result = {
        'shifted_test': test_dv[new_test_mask],
        'non_shifted_test': test_dv[old_test_mask]
    }        
    new_data = Log.get_log_data(acquisition_config, model_config, dataset_splits)
    train_data_loader = torch.utils.data.DataLoader(new_data, batch_size=model_config.batch_size, 
                            num_workers= Dataset.config['num_workers'])
    train_dv, _ = product.clf.predict(train_data_loader)    
    dv_result['shifted_train'] = train_dv
    return dv_result

def run(acquisition_config:Config.Acquistion, model_config:Config.NewModel, method, n_data, product:Checker.subset, data_splits:Dataset.DataSplits):
    acquisition_config.set_items(method,n_data)
    bound = Subset.get_threshold(product.clf, acquisition_config, model_config, data_splits)
    dv = compare(acquisition_config, model_config, product, data_splits, bound)
    old_train_dv, _ = product.clf.predict(data_splits.loader['train_clip'])
    dv['non_shifted_train'] = old_train_dv
    return dv

def main(epochs, new_model_setter='retrain', model_dir ='', ac_methods='', ac_number=0, device=0):
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device)
    results = []

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        
        old_model_config = Config.OldModel(batch_size,superclass_num,model_dir, device_config, epo)
        new_model_config = Config.NewModel(batch_size,superclass_num,model_dir, device_config, epo, True, new_model_setter, False)
        acquire_instruction = Config.AcquistionFactory('seq',seq_rounds_config) 
        dataset = ds_list[epo]
        dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)

        product = Checker.factory('ts', new_model_config)
        product.setup(old_model_config, dataset_splits)

        result_epoch = run( acquire_instruction, new_model_config, ac_methods, ac_number, product, dataset_splits)
        results.append(result_epoch)

    result_plotter = Plotter.Histogram(new_model_config)
    result_plotter.run(epochs, results, ac_number, ac_methods)  

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-p','--pure',type=Config.str2bool,default=False)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-am','--ac_methods',type=str, default='dv')
    parser.add_argument('-an','--ac_number',type=int, default=200)
    parser.add_argument('-d','--device',type=int,default=0)

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, ac_methods=args.ac_methods, ac_number=args.ac_number, device=args.device)