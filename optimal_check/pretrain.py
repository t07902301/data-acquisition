import sys
sys.path.append('/home/yiwei/data-acquisition')
from set_up import *
from utils.strategy import *
from utils.logging import *

def run(dataset, model_config:Config.OptModel, train_flag:bool, parse_args, src_model_config:Config.OldModel):
    config, dataset_name, _ = parse_args
    if train_flag:
        ds = dataset_utils.DataSplits(dataset, model_config.batch_size, dataset_name)
        optimal_model = Model.CNN(config['hparams']['optimal'])
        optimal_model.train(ds.loader['train'], ds.loader['val_shift'], config['hparams']['optimal'])
    else:
        ds = dataset_utils.DataSplits(dataset, config['hparams']['optimal']['batch_size']['new'], dataset_name)
        optimal_model = Model.CNN(config['hparams']['optimal'])
        optimal_model.load(model_config.path, model_config.device)
    
    source_model = Model.factory(src_model_config.model_type, config)
    source_model.load(src_model_config.path, src_model_config.device)
    src_acc = source_model.acc(ds.loader['test_shift'])

    # Evaluate
    opt_acc = optimal_model.acc(ds.loader['test_shift'])
    logger.info('Test Shift Acc: {}'.format(opt_acc))

    if train_flag:
        optimal_model.save(model_config.path)

    return opt_acc, opt_acc - src_acc

def main(epochs,  model_dir, train_flag, device_id, base_type, detector_name):
    fh = logging.FileHandler('log/{}/opt.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info('Detector Name: {}'.format(detector_name))
    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device_id)
    parse_args = (config, dataset_name, option)
    opt_acc_list, acc_change_list = [], []
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        opt_model_config = Config.OptModel(config['hparams']['optimal']['batch_size']['base'], config['hparams']['optimal']['superclass'], model_dir, device_config, epo, base_type)
        src_model_config = Config.OldModel(config['hparams']['source']['batch_size']['base'], config['hparams']['source']['superclass'], model_dir, device_config, epo, base_type)
        ds = ds_list[epo]
        opt_acc, acc_change = run(ds, opt_model_config, train_flag, parse_args,src_model_config)
        opt_acc_list.append(opt_acc)
        acc_change_list.append(acc_change)
    logger.info('Old Model Acc after shift: {}%'.format(np.round(np.mean(opt_acc_list),decimals=3)))
    logger.info('Src Model Acc Change: {}%'.format(np.round(np.mean(acc_change_list),decimals=3)))

    for spit_name in ds.keys():
        logger.info('{}: {}'.format(spit_name, len(ds[spit_name])))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name) _ task _ (other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-tf','--train_flag',type=bool,default=False, help='train or test source model')

    args = parser.parse_args()
    main(args.epochs,args.model_dir,args.train_flag, args.device, args.base_type, args.detector_name)

