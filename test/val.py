import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))

from utils.strategy import *
from utils.set_up import *
import utils.statistics.distribution as distribution_utils
from typing import List
import utils.dataset.wrappers as dataset_utils
from abc import abstractclassmethod

batch_size = 32 # the size of validation samples

class Checker():
    def __init__(self) -> None:
        pass

    def get_test_info(self, dataset, dataloader, clf:Detector.Prototype, batch_size):
        data_info = {}
        dv, _ = clf.predict(dataloader)        
        data_info['dv'] = dv
        data_info['old_batch_size'] = batch_size
        data_info['new_batch_size'] = batch_size
        data_info['dataset'] = dataset
        return data_info
    
    def set_up_dstr(self, base_model, set_up_loader, pdf, clf):
        correct_dstr = distribution_utils.CorrectnessDisrtibution(base_model, clf, set_up_loader, pdf, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(base_model, clf, set_up_loader, pdf, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}
    
    def test_dstr_probab(self, base_model, set_up_loader, test_info, clf, stream: Config.ProbabStream):
        dstr = self.set_up_dstr(base_model, set_up_loader, stream.pdf, clf)
        test_loader, _ = Partitioner.Probability().run(test_info, {'target': dstr['incorrect'], 'other': dstr['correct']}, stream)
        return test_loader

    def _target_test(self, structural_loader, new_model:Model.Prototype):
        '''
        loader: new_model + old_model
        '''
        if len(structural_loader['old_model']) == 0:
            gt,pred,_  = new_model.eval(structural_loader['new_model'])
            new_correct = (gt==pred)
            logger.info('Nothing for Old Model')
            total_correct = new_correct
        elif len(structural_loader['new_model']) == 0:
            logger.info('Nothing for New Model')
            gt,pred,_  = self.base_model.eval(structural_loader['old_model'])
            old_correct = (gt==pred)
            total_correct = old_correct
        else:
            gt,pred,_  = new_model.eval(structural_loader['new_model'])
            new_correct = (gt==pred)
            gt,pred,_  = self.base_model.eval(structural_loader['old_model'])
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
        
        return total_correct.mean()*100 
        # return new_correct.mean() * 100
    
    def set_up(self, test_dataset, test_loader, clf:Detector.Prototype, base_model:Model.Prototype, batch_size, set_up_loader, stream):
        test_info = self.get_test_info(test_dataset, test_loader, clf, batch_size)
        self.test_structural_loader = self.test_dstr_probab(base_model, set_up_loader, test_info, clf, stream) 
        self.base_model = base_model
        self.detector = clf

    def run(self, new_model):
        return self._target_test(self.test_structural_loader, new_model)

# class Strategy():
#     def __init__(self) -> None:
#         pass

#     @abstractclassmethod
#     def acquistion(self, n_samples, dataset, target_labels):
#         pass

#     def get_new_model(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, val_loader):
#         train_loader = torch.utils.data.DataLoader(train_data, batch_size)
#         new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)
#         if model_config.base_type == 'cnn':
#             new_model.train(train_loader, val_loader)
#         else:
#             new_model.train(train_loader)
#         return new_model

#     @abstractclassmethod
#     def get_sample_performance(self, samples_indices, model_config, data_split: dataset_utils.DataSplits, detect_instruction, config, base_acc):
#         pass

class Random_Sample():
    def __init__(self) -> None:
        pass

    # def acquistion(self, n_samples, val_size):
    #     val_indices = np.arange(val_size)

    #     samples_indices = {}
    #     for size in n_samples:
    #         np.random.seed(0)
    #         samples_indices[size] = np.random.choice(val_indices, size, replace=False)
    #     return samples_indices

    def acquistion(self, n_samples, dataset, target_labels):
        samples_indices = {}
        tool = dataset_utils.Cifar()
        dataset_labels = tool.get_labels(dataset['val_shift'], use_fine_label=True)
        dataset_size = len(dataset_labels)
        for size in n_samples:       
            np.random.seed(0)
            samples_indices[size], _ = tool.get_split_indices(dataset_labels, target_labels, size/dataset_size)
        return samples_indices

    def get_model_performance(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, test_loader, val_loader):

        train_loader = torch.utils.data.DataLoader(train_data, batch_size)

        new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)

        if model_config.base_type == 'cnn':
            new_model.train(train_loader, val_loader)
        else:
            new_model.train(train_loader)

        new_acc = new_model.acc(test_loader)

        return new_acc
    
    def get_sample_performance(self, samples_indices, model_config, data_split: dataset_utils.DataSplits, detect_instruction, config, base_acc):

        pairs = []

        for n_sample, indices in samples_indices.items():

            val_sample = torch.utils.data.Subset(data_split.dataset['val_shift'], indices)

            new_acc = self.get_model_performance(model_config, detect_instruction, config, val_sample, data_split.loader['test_shift'], data_split.loader['val_reg'])

            pairs.append((n_sample, new_acc - base_acc))

        return pairs

    def run(self, n_samples, epochs, ds_list:List[dataset_utils.Dataset], config, device_config, base_type, model_dir, detect_instrution):

        sample_src_size = len(ds_list[0]['val_shift'])

        regression_pairs_dict = {}

        for epo in range(epochs):

            # sample_indices = self.acquistion(n_samples, sample_src_size)

            logger.info('in epoch {}'.format(epo))
            model_config = Config.OldModel(config['hparams']['batch_size']['base'], config['hparams']['superclass'], model_dir, device_config, epo, base_type)        
            ds = ds_list[epo]

            sample_indices = self.acquistion(n_samples, ds, [i for i in range(100)])

            ds = dataset_utils.DataSplits(ds, model_config.batch_size)

            base_model = Model.factory(model_config.base_type, config, detect_instrution.vit)
            base_model.load(model_config.path, model_config.device)
            base_acc = base_model.acc(ds.loader['test_shift'])

            regression_pairs = self.get_sample_performance(sample_indices, model_config, ds,
                                                           detect_instrution, config, base_acc)

            regression_pairs_dict[epo] = regression_pairs
        
        return regression_pairs_dict

class Greedy():
    def __init__(self) -> None:
        pass

    def acquisition(self, n_samples, dataloader, detector: Detector.Prototype, base_model: Model.Prototype, dataset):

        dv, _ = detector.predict(dataloader, base_model)

        samples_indices = {}
        for size in n_samples:
            samples_indices[size] = acquistion.get_top_values_indices(dv, size)

        return samples_indices

    # def get_model_performance_dev(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, test_loader):

    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size)

    #     new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)
    #     new_model.train(train_loader)

    #     new_acc = new_model.acc(test_loader)

    #     return new_acc

    def get_model_performance(self, model_type, detect_instruction:Config.Detection, config, train_data, checker:Checker, val_loader):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size)
        new_model = Model.factory(model_type, config, detect_instruction.vit)

        if model_type == 'cnn':
            new_model.train(train_loader, val_loader)
        else:
            new_model.train(train_loader)

        new_acc = checker.run(new_model)
        return new_acc
    
    def get_sample_probab(self, sample_dv, distribution: distribution_utils.Disrtibution):
        return distribution.pdf.accumulate(sample_dv['max'], sample_dv['min'])
    
    def get_dv_range(self, sample_data, checker:Checker):
        sample_loader = torch.utils.data.DataLoader(sample_data, batch_size)
        dv, _ = checker.detector.predict(sample_loader, checker.base_model)
        # logger.info('in get range:', checker.base_model.acc(sample_loader))
        # logger.info(len(dv), max(dv), min(dv))
        return {
            'max': max(dv),
            'min': min(dv)
        }
    
    def get_sample_performance(self, samples_indices:dict, model_config:Config.OldModel, detect_instruction:Config.Detection, config, data_split: dataset_utils.DataSplits, checker:Checker, base_acc):

        distribution = distribution_utils.Disrtibution(checker.detector, data_split.loader['val_shift'], 'kde')

        pairs = []

        for n_sample, indices in samples_indices.items():
            val_sample = torch.utils.data.Subset(data_split.dataset['val_shift'], indices)
            new_acc = self.get_model_performance(model_config.base_type, detect_instruction, config, val_sample, checker, data_split.loader['val_shift'])
            # new_acc = self.get_model_performance_dev(model_config, detect_instruction, config, val_sample, data_split.loader['test_shift'])
            val_sample_dv = self.get_dv_range(val_sample, checker)
            pairs.append((self.get_sample_probab(val_sample_dv, distribution), new_acc - base_acc, n_sample))
            # pairs.append((n_sample, new_acc - base_acc))

        return pairs
    
    def get_dstr_clf(self, detect_instrution, config, train_loader, base_model):
        clf = Detector.factory(detect_instrution.name, config, clip_processor = detect_instrution.vit, split_and_search=True, data_transform='clip')
        _ = clf.fit(base_model, train_loader) 
        return clf
    
    def run(self, n_samples, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution):
        regression_pairs_dict = {}
        stream = Config.ProbabStream(bound=0.5, pdf='kde', name='probab')

        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            model_config = Config.OldModel(config['hparams']['batch_size']['base'], config['hparams']['superclass'], model_dir, device_config, epo, base_type)        
            ds = ds_list[epo]
            ds = dataset_utils.DataSplits(ds, batch_size)

            base_model = Model.factory(model_config.base_type, config, detect_instrution.vit)
            base_model.load(model_config.path, model_config.device)
            base_acc = base_model.acc(ds.loader['test_shift'])

            clf = self.get_dstr_clf(detect_instrution, config, ds.loader['val_shift'], base_model)

            sample_indices = self.acquisition(n_samples, ds.loader['val_shift'], clf, base_model, ds.dataset['val_shift'])

            checker = Checker()
            checker.set_up(ds.dataset['test_shift'], ds.loader['test_shift'], clf, base_model, batch_size, ds.loader['val_shift'], stream)

            regression_pairs = self.get_sample_performance(sample_indices, model_config, detect_instrution, config, ds, checker, base_acc)

            regression_pairs_dict[epo] = regression_pairs
        
        return regression_pairs_dict 

def export(model_dir, dev, data):
   
    file = os.path.join('log/{}/reg'.format(model_dir), '{}.pkl'.format(dev))
   
    with open(file, 'wb') as f:
        out = pkl.dump(data, f)

    logger.info('save to {}'.format(file))

    return out

def main(epochs,  model_dir ='', device_id=0, base_type='', detector_name='', dev = '', option = '', dataset_name = ''):

    fh = logging.FileHandler('log/{}/val_{}.log'.format(model_dir, dev), mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info('Detector Name: {}'.format(detector_name))
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device_id, option, dataset_name)
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    detect_instrution = Config.Detection(detector_name, clip_processor)

    # np.random.seed(0)
    # sample_size = np.random.choice(np.arange(50, len(ds_list[0]['val_shift'])), 30, replace=False)

    # logger.info(sorted(sample_size))

    # sample_size = [i for i in range(15, len(ds_list[0]['val_shift'])+1, 5)]

    sample_size = [i for i in range(400, len(ds_list[0]['val_shift'])+1, 400)]

    logger.info('samples: {}'.format(sample_size))
    # return

    if dev == 'rs':
        rs = Random_Sample()
        regression_pairs_dict = rs.run(sample_size, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution)
    else:
        greedy = Greedy()
        regression_pairs_dict = greedy.run(sample_size, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution)

    export(model_dir, dev, regression_pairs_dict)
    # plot_regress(epochs, regression_pairs_dict, model_dir, dev)
    # logger.info(regression_pairs_dict)

    split_data = ds_list[0]
    for spit_name in split_data.keys():
        logger.info('{}:{}'.format(spit_name, len(split_data[spit_name])))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-ds','--dataset',type=str, default='core')
    parser.add_argument('-op','--option',type=str, default='object')
    args = parser.parse_args()
    main(args.epochs,args.model_dir, args.device, args.base_type, args.detector_name, args.dev)