from utils.strategy import *
from utils.set_up import *
import utils.statistics.data as DataStat
from utils.logging import *
import matplotlib.pyplot as plt

def run(dataset, model_config:Config.OldModel, train_flag:bool, detect_instruction:Config.Detection, parse_args:ParseArgs, epoch):
    if train_flag:
        ds = dataset_utils.DataSplits(dataset, model_config.batch_size, parse_args.dataset_name)
        if model_config.model_type == 'cnn':
            base_model = Model.CNN(parse_args.general_config['hparams']['source'])
            base_model.train(ds.loader['train'], ds.loader['val'], parse_args.general_config['hparams']['source'])
        else:
            base_model = Model.svm(parse_args.general_config['detector_args'], detect_instruction.vit)
            base_model.train(ds.loader['train_non_cnn'])
    else:
        ds = dataset_utils.DataSplits(dataset, parse_args.general_config['hparams']['source']['batch_size']['new'], parse_args.dataset_name)
        base_model = Model.factory(model_config.model_type, parse_args.general_config, detect_instruction.vit)
        base_model.load(model_config.path, model_config.device)

    # Evaluate
    acc = base_model.acc(ds.loader['test'])
    acc_shift = base_model.acc(ds.loader['test_shift'])
    val_acc_shift = base_model.acc(ds.loader['val_shift'])
    logger.info('Test Shift Acc: {}'.format (acc_shift))

    # _, _, probabs = base_model.eval(ds.loader['val_shift'])
    # entropy = -np.sum(probabs * np.log2(probabs), axis=1)
    # # entropy = entropy.reshape((len(entropy), 1))
    # plt.hist(entropy, 20)
    # plt.savefig('figure/etp/{}.png'.format(epoch))
    # plt.cla()
    # return 0,0,0,0,0


    detector = Detector.factory(detect_instruction.name, parse_args.general_config, clip_processor = detect_instruction.vit, split_and_search=True, data_transform='clip')
    detector.fit(base_model, ds.loader['val_shift'], detect_instruction.weaness_label_generator)
    
    _, detect_prec = detector.predict(ds.loader['test_shift'], metrics='precision', base_model=base_model)
    logger.info('In testing Detector: {}'.format(detect_prec))

    if train_flag:
        base_model.save(model_config.path)

    shift_score = DataStat.error_label_stat('test_shift', ds, base_model, parse_args.general_config['data']['labels']['remove'], parse_args.task)

    return acc, acc_shift, detect_prec, shift_score, val_acc_shift

def main(epochs,  model_dir, train_flag, device, detector_name, weaness_label_generator):
    fh = logging.FileHandler('log/{}/base.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info('Detector Name: {}'.format(detector_name))
    parse_args, dataset_list, normalize_stat = set_up(epochs, model_dir, device)
    acc_list, acc_shift_list, detect_prec_list, shift_list, val_acc_shift_list = [], [], [], [], []
    clip_processor = Detector.load_clip(parse_args.device_config, normalize_stat['mean'], normalize_stat['std'])
    detect_instrution = Config.Detection(detector_name, clip_processor, weaness_label_generator)
    for epo in range(epochs):
        logger.info('in epoch {}'.format(epo))
        old_model_config, _, _ = parse_args.get_model_config(epo)
        ds = dataset_list[epo]
        acc, acc_shift, detect_prec, shift_score, val_shift = run(ds, old_model_config, train_flag, detect_instrution, parse_args, epo)
        acc_list.append(acc)
        acc_shift_list.append(acc_shift)
        detect_prec_list.append(detect_prec)
        shift_list.append(shift_score)
        val_acc_shift_list.append(val_shift)
    logger.info('Old Model Acc before shift: {}%'.format(np.round(np.mean(acc_list),decimals=3)))
    logger.info('Old Model Acc after shift: {}%'.format(np.round(np.mean(acc_shift_list),decimals=3)))
    logger.info('Shifted Data Proportion on Old Model Misclassifications: {}%'.format(np.round(np.mean(np.array(shift_list),axis=0), decimals=3)))
    logger.info('Detector Average Metrics: {}%'.format(np.round(np.mean(detect_prec_list), decimals=3).tolist()))
    logger.info('Old Model Error in Val Shift: {}%'.format(100 - np.round(np.mean(val_acc_shift_list), decimals=3)))

    for spit_name in ds.keys():
        logger.info('{}: {}'.format(spit_name, len(ds[spit_name])))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name) _ task _ (other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")
    parser.add_argument('-tf','--train_flag',type=bool,default=False, help='train or test source model')
    parser.add_argument('-wlg','--weaness_label_generator',type=str,default='correctness', help="correctness, entropy+gmm")

    args = parser.parse_args()
    main(args.epochs,args.model_dir,args.train_flag, args.device, args.detector_name, args.weaness_label_generator)

