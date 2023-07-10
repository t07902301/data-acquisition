from copy import deepcopy
from abc import abstractmethod
import utils.log as Log
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
import torch
import numpy as np
import utils.statistics.subset as TestSubset
import utils.statistics.stat_test as Plot_Stat
class Strategy():
    base_model:Model.prototype
    def __init__(self, old_model_config:Config.OldModel, clip_processor=None) -> None:
        self.base_model = Model.prototype_factory(old_model_config.base_type, old_model_config.class_number, clip_processor)
        self.base_model.load(old_model_config)

    @abstractmethod
    def operate(self, acquire_instruction: Config.Acquistion, dataset: dict, new_model_config:Config.NewModel):
        '''
        Under some acquistion instruction (which method, how much new data), this strategy operates on data and the old Model. Finally, the new model will be saved.
        '''
        pass
    @abstractmethod
    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.Acquistion):
        '''
        Log data to reconstruct the final train set, and the final CLF statistic if necessary
        '''
        pass

    def get_new_val(self, dataset_splits: Dataset.DataSplits, stream_instruction:Config.ProbabStream, model_config: Config.NewModel, clf=None, detect_instruction:Config.Dectector=None):
        print('Get split val_shift')
        if clf is None and detect_instruction != None:
            clf = Detector.factory(detect_instruction.name, clip_processor = detect_instruction.vit, split_and_search=True)
            _ = clf.fit(self.base_model, dataset_splits.loader['val_shift']) 
        # val_dv, _ = clf.predict(dataset_splits.loader['val_shift'], self.base_model)
        # dataset_splits.replace('new_data', new_data)
        # new_data_dv, _ = clf.predict(dataset_splits.loader['new_data'], self.base_model)
        # bound = np.max(new_data_dv)
        # val_indices = np.arange(len(dataset_splits.dataset['val_shift']))
        # targeted_indices = val_indices[val_dv <= bound]
        # targeted_val = torch.utils.data.Subset(dataset_splits.dataset['val_shift'], targeted_indices)
        # dataset_splits.replace('val_shift', targeted_val)   

        cor_dv, incor_dv = Plot_Stat.get_dv_dstr(self.base_model, dataset_splits.loader['val_shift'], clf)
        correct_prior = (len(cor_dv)) / (len(cor_dv) + len(incor_dv))
        correct_dstr = TestSubset.disrtibution(correct_prior, Plot_Stat.get_pdf(cor_dv, stream_instruction.pdf))
        incorrect_dstr =  TestSubset.disrtibution(1 - correct_prior, Plot_Stat.get_pdf(incor_dv, stream_instruction.pdf))
        val_shift_info = TestSubset.build_data_info(dataset_splits, 'val_shift', clf, model_config, self.base_model)
        val_shift_split, _ = TestSubset.probability_setter().get_subset_loders(val_shift_info, correct_dstr, incorrect_dstr, stream_instruction)
        dataset_splits.loader['val_shift'] = val_shift_split['new_model']
        dataset_splits.dataset['val_shift'] = None

    def get_new_data(self, data_splits: Dataset.DataSplits, new_data_indices):
        return torch.utils.data.Subset(data_splits.dataset['market'],new_data_indices)
    
class NonSeqStrategy(Strategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
    @abstractmethod
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Dectector = None, bound = None):
        '''
        Based on old model performance, a certain number of new data is acquired.
        '''
        pass

    def operate(self, acquire_instruction: Config.Acquistion, dataset: dict, new_model_config:Config.NewModel):
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, new_model_config.new_batch_size)
        new_data_indices,_ = self.get_new_data_indices(acquire_instruction.n_ndata, dataset_splits, acquire_instruction.detector, acquire_instruction.bound)
        new_data = self.get_new_data(dataset_splits, new_data_indices)
        dataset_splits.use_new_data(new_data, new_model_config, acquire_instruction)

        self.log_data(new_model_config, new_data_indices, acquire_instruction)
       
        self.get_new_val(dataset_splits, acquire_instruction.stream, new_model_config, detect_instruction=acquire_instruction.detector)

        self.base_model.update(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'])
        new_model_config.set_path(acquire_instruction)
        self.base_model.save(new_model_config.path)

    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.Acquistion):
        idx_log = model_config.get_log_config('indices')
        idx_log.set_path(acquire_instruction)
        Log.save(data, idx_log)

class Greedy(NonSeqStrategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Dectector, bound = None):
        clf = Detector.factory(detector_instruction.name, clip_processor = detector_instruction.vit, split_and_search=True)
        score = clf.fit(self.base_model, dataset_splits.loader['val_shift'], dataset_splits.dataset['val_shift'], 16)
        market_dv, _ = clf.predict(dataset_splits.loader['market'], self.base_model)
        if bound == None:
            new_data_indices = acquistion.get_top_values_indices(market_dv, n_data)
        else:
            new_data_indices = acquistion.get_in_bound_top_indices(market_dv, n_data, bound)
        clf_info = {
            'clf': clf,
            'score': score
        }
        return new_data_indices, clf_info       

class Sample(NonSeqStrategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Dectector, bound = None):
        market_gts = acquistion.get_loader_labels(dataset_splits.loader['market'])
        new_data_indices = acquistion.sample_acquire(market_gts,n_data)
        clf_info = None
        return new_data_indices, clf_info      

class Confidence(NonSeqStrategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
        self.n_class = old_model_config.class_number
    
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Dectector, bound = None):
        market_gts, _, market_confs = self.base_model.eval(dataset_splits.loader['market'])
        if self.n_class == 1:
            confs_diff = acquistion.get_conf_diff(market_gts, market_confs)
            new_data_indices = acquistion.get_top_values_indices(confs_diff, n_data)
        else:
            new_data_indices = acquistion.get_top_values_indices(market_confs, n_data)
        clf_info = None
        return new_data_indices, clf_info       

class Mix(NonSeqStrategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Dectector, bound = None):
        clf = Detector.SVM(dataset_splits.loader['train_clip'], self.clip_processor)
        _ = clf.fit(self.base_model, dataset_splits.loader['val_shift'])
        market_dv, _ = clf.predict(dataset_splits.loader['market'], self.base_model)
        greedy_results = acquistion.get_top_values_indices(market_dv, n_data-n_data//2)
        sample_results = acquistion.sample_acquire(market_dv,n_data//2)
        new_data_cls_indices = np.concatenate([greedy_results, sample_results])
        clf_info = None
        return new_data_cls_indices, clf_info       

class SeqCLF(Strategy):
    def __init__(self, old_model_config: Config.OldModel) -> None:
        super().__init__(old_model_config)
        self.base_model_config = old_model_config
   
    def operate(self, acquire_instruction:Config.SequentialAc, dataset:dict, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method, self.base_model_config)
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, new_model_config.new_batch_size)
        org_val_ds = dataset_splits.dataset['val_shift']
        new_data_total_set = None
        rounds = 2

        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, dataset_splits)
            new_data_round_set = torch.utils.data.Subset(dataset_splits.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(dataset_splits, new_data_round_indices, )

            dataset_splits.reduce('market', new_data_round_indices)
            dataset_splits.expand('val_shift', new_data_round_set)

            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        # recover original val 
        dataset_splits.use_new_data(new_data_total_set, new_model_config, acquire_instruction)
        assert len(org_val_ds) == len(dataset_splits.dataset['val_shift']) - acquire_instruction.n_ndata, "size error with original val"
        dataset_splits.replace('val_shift', org_val_ds)
        # train model 
        self.get_new_val(dataset_splits, new_data_total_set, clf_info['clf'])
        self.base_model.update(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'])
        new_model_config.set_path(acquire_instruction)
        self.base_model.save(new_model_config.path)

        clf = clf_info['clf']
        self.log_data(new_model_config, new_data_total_set, acquire_instruction,clf)

    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.SequentialAc, detector):
        data_config = model_config.get_log_config('data')
        data_config.set_path(acquire_instruction)
        Log.save(data, data_config) # Save new data
        clf_config = model_config.get_log_config('clf')
        clf_config.set_path(acquire_instruction)
        Log.save(detector.fitter.clf, clf_config) # Save new clf

class Seq(Strategy):
    def __init__(self) -> None:
        super().__init__()
    def operate(self, acquire_instruction:Config.SequentialAc, dataset: dict, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, old_model_config.batch_size)
        new_data_total_set = None
        rounds = 2
        model = Model.load(old_model_config)
        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, dataset_splits, old_model_config)
            new_data_round_set = torch.utils.data.Subset(dataset_splits.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(dataset_splits, new_data_round_indices, )

            dataset_splits.reduce('market', new_data_round_indices)
            dataset_splits.expand('train_clip', new_data_round_set)
            dataset_splits.expand('train', new_data_round_set)

            model = Model.get_new(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'], model)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        assert len(new_data_total_set) == acquire_instruction.n_ndata, 'size error with new data'
        if new_model_config.pure:
            dataset_splits.replace('train', new_data_total_set)
            model = Model.get_new(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'], model)

        new_model_config.set_path(acquire_instruction)
        Model.save(model,new_model_config.path)

        clf = clf_info['clf']
        self.log_data(new_model_config, new_data_total_set, acquire_instruction, clf)

    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.SequentialAc, clf):
        data_config = Log.get_config(model_config, acquire_instruction, 'data')
        Log.save(data, data_config) # Save new data
        clf_config = Log.get_config(model_config, acquire_instruction, 'clf')
        Log.save(clf, clf_config) # Save new clf

def StrategyFactory(strategy, old_model_config:Config.OldModel):
    if strategy=='dv':
        return Greedy(old_model_config)
    elif strategy =='sm':
        return Sample(old_model_config)
    elif strategy == 'conf':
        return Confidence(old_model_config)
    elif strategy == 'mix':
        return Mix(old_model_config)
    elif strategy == 'seq':
        return Seq()
    else:
        return SeqCLF(old_model_config)