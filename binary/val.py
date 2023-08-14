from utils.strategy import *
from utils.set_up import *

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
        correct_dstr = Distribution.get_correctness_dstr(base_model, clf, set_up_loader, pdf, correctness=True)
        incorrect_dstr = Distribution.get_correctness_dstr(base_model, clf, set_up_loader, pdf, correctness=False)
        return {'correct': correct_dstr, 'incorrect': incorrect_dstr}
    
    def test_dstr_probab(self, base_model, set_up_loader, test_info, clf, stream: Config.ProbabStream):
        dstr = self.set_up_dstr(base_model, set_up_loader, stream.pdf, clf)
        test_loader, _ = Partitioner.Probability().run(test_info, {'target': dstr['incorrect'], 'other': dstr['correct']}, stream)
        return test_loader

    def _target_test(self, structural_loader, new_model:Model.prototype):
        '''
        loader: new_model + old_model
        '''
        gt,pred,_  = new_model.eval(structural_loader['new_model'])
        new_correct = (gt==pred)

        if len(structural_loader['old_model']) == 0:
            total_correct = new_correct

        else:
            gt,pred,_  = self.base_model.eval(structural_loader['old_model'])
            old_correct = (gt==pred)
            total_correct = np.concatenate((old_correct,new_correct))
        
        return total_correct.mean()*100 
        # return new_correct.mean() * 100
    
    def set_up(self, test_dataset, test_loader, clf, base_model:Model.prototype, batch_size, set_up_loader, stream):
        test_info = self.get_test_info(test_dataset, test_loader, clf, batch_size)
        self.test_structural_loader = self.test_dstr_probab(base_model, set_up_loader, test_info, clf, stream) 
        self.base_model = base_model

    def run(self, new_model):
        return self._target_test(self.test_structural_loader, new_model)

class Random_Sample():
    def __init__(self) -> None:
        pass

    def acquistion(self, n_samples, val_size):
        np.random.seed(0)
        val_indices = np.arange(val_size)

        samples_indices = {}
        for size in n_samples:
            samples_indices[size] = np.random.choice(val_indices, size, replace=False)
        return samples_indices

    def get_model_performance(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, test_loader):

        train_loader = torch.utils.data.DataLoader(train_data, 16)

        new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)
        new_model.train(train_loader)

        new_acc = new_model.acc(test_loader)

        return new_acc
    
    def get_sample_performance(self, samples_indices, model_config, data_split: Dataset.DataSplits, detect_instruction, config, base_acc):

        pairs = []

        for n_sample, indices in samples_indices.items():

            val_sample = torch.utils.data.Subset(data_split.dataset['val_shift'], indices)

            new_acc = self.get_model_performance(model_config, detect_instruction, config, val_sample, data_split.loader['test_shift'])

            pairs.append((n_sample, new_acc - base_acc))

        return pairs

    def run(self, n_samples, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution):

        sample_indices = self.acquistion(n_samples, len(ds_list[0]['val_shift']))

        regression_pairs_dict = {}

        for epo in range(epochs):
            print('in epoch {}'.format(epo))
            model_config = Config.OldModel(config['hparams']['batch_size']['base'], config['hparams']['superclass'], model_dir, device_config, epo, base_type)        
            ds = ds_list[epo]
            ds = Dataset.DataSplits(ds, model_config.batch_size)

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

    def acquisition(self, n_samples, val_loader, detector: Detector.Prototype, base_model):

        dv, _ = detector.predict(val_loader, base_model)

        samples_indices = {}
        for size in n_samples:
            samples_indices[size] = acquistion.get_top_values_indices(dv, size)

        return samples_indices

    def get_model_performance_dev(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, test_loader):

        train_loader = torch.utils.data.DataLoader(train_data, 16)

        new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)
        new_model.train(train_loader)

        new_acc = new_model.acc(test_loader)

        return new_acc

    def get_model_performance(self, model_config:Config.OldModel, detect_instruction:Config.Detection, config, train_data, checker:Checker):
        train_loader = torch.utils.data.DataLoader(train_data, 16)
        new_model = Model.factory(model_config.base_type, config, detect_instruction.vit)
        new_model.train(train_loader)
        new_acc = checker.run(new_model)
        return new_acc
    
    def get_sample_performance(self, samples_indices:dict, model_config:Config.OldModel, detect_instruction:Config.Detection, config, data_split: Dataset.DataSplits, checker:Checker, base_acc):

        pairs = []

        for n_sample, indices in samples_indices.items():
            val_sample = torch.utils.data.Subset(data_split.dataset['val_shift'], indices)
            new_acc = self.get_model_performance(model_config, detect_instruction, config, val_sample, checker)
            # new_acc = self.get_model_performance_dev(model_config, detect_instruction, config, val_sample, data_split.loader['test_shift'])
            pairs.append((n_sample, new_acc - base_acc))
            # pairs.append((n_sample, new_acc))

        return pairs
    
    def get_dstr_clf(self, detect_instrution, config, train_loader, base_model):
        clf = Detector.factory(detect_instrution.name, config, clip_processor = detect_instrution.vit, split_and_search=True)
        _ = clf.fit(base_model, train_loader) 
        return clf
    
    def run(self, n_samples, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution):
        regression_pairs_dict = {}
        stream = Config.ProbabStream(bound=0.5, pdf='kde', name='probab')

        for epo in range(epochs):
            print('in epoch {}'.format(epo))
            model_config = Config.OldModel(config['hparams']['batch_size']['base'], config['hparams']['superclass'], model_dir, device_config, epo, base_type)        
            ds = ds_list[epo]
            ds = Dataset.DataSplits(ds, model_config.batch_size)

            base_model = Model.factory(model_config.base_type, config, detect_instrution.vit)
            base_model.load(model_config.path, model_config.device)
            base_acc = base_model.acc(ds.loader['test_shift'])

            clf = self.get_dstr_clf(detect_instrution, config, ds.loader['val_shift'], base_model)

            sample_indices = self.acquisition(n_samples, ds.loader['val_shift'], clf, base_model)

            checker = Checker()
            checker.set_up(ds.dataset['test_shift'], ds.loader['test_shift'], clf, base_model, 16, ds.loader['val_shift'], stream)

            regression_pairs = self.get_sample_performance(sample_indices, model_config, detect_instrution, config, ds, checker, base_acc)

            regression_pairs_dict[epo] = regression_pairs
        
        return regression_pairs_dict 

def export(model_dir, dev, data):
   
    file = os.path.join('log/{}/dev'.format(model_dir), 'val_{}.pkl'.format(dev))
   
    with open(file, 'wb') as f:
        out = pkl.dump(data, f)

    print('save to', file)

    return out

def main(epochs,  model_dir ='', device_id=0, base_type='', detector_name='', dev = ''):

    print('Detector Name:', detector_name)
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device_id)
    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    detect_instrution = Config.Detection(detector_name, clip_processor)

    np.random.seed(0)
    sample_size = np.random.choice(np.arange(50, len(ds_list[0]['val_shift'])), 75, replace=False)
    # sample_size = [i for i in range(30, len(ds_list[0]['val_shift']), 10)]

    if dev == 'rs':
        rs = Random_Sample()
        regression_pairs_dict = rs.run(sample_size, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution)
    else:
        greedy = Greedy()
        regression_pairs_dict = greedy.run(sample_size, epochs, ds_list, config, device_config, base_type, model_dir, detect_instrution)

    export(model_dir, dev, regression_pairs_dict, )

    split_data = ds_list[0]
    for spit_name in split_data.keys():
        print(spit_name, len(split_data[spit_name]))

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')

    args = parser.parse_args()
    main(args.epochs,args.model_dir, args.device, args.base_type, args.detector_name, args.dev)