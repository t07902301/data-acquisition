from copy import deepcopy
from abc import abstractmethod

import utils.log as Log
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import utils.objects.CLF as CLF
import torch
import numpy as np
from utils import config
class Strategy():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def operate(self, acquire_instruction: Config.Acquistion, dataset: dict, old_model_config:Config.OldModel, new_model_config:Config.NewModel):
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
class NonSeqStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
    @abstractmethod
    def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, old_model_config:Config.OldModel):
        '''
        Based on old model performance, a certain number of new data is acquired.
        '''
        pass
    def operate(self, acquire_instruction: Config.Acquistion, dataset: dict, old_model_config:Config.OldModel, new_model_config:Config.NewModel):
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, old_model_config.batch_size)
        new_data_indices,_ = self.get_new_data_indices(acquire_instruction.new_data_number_per_class, dataset_splits, old_model_config)
        new_data_set = self.get_new_data(dataset_splits, new_data_indices, new_model_config.augment)
        dataset_splits.use_new_data(new_data_set, new_model_config, acquire_instruction)

        self.log_data(new_model_config, new_data_indices, acquire_instruction)

        new_model = Model.get_new(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'])
        new_model_config.set_path(acquire_instruction)
        Model.save(new_model,new_model_config.path)

        # base_model = Model.load(old_model_config)
        # clf = CLF.SVM(ds.loader['train_clip'])
        # score = clf.fit(base_model, ds.loader['val_shift'])
        # market_info, precision = clf.predict(ds.loader['market'])
        # for cls_idx in raw_new_data_indices:
        #     print(np.max(market_info['dv'][cls_idx]))
        # new_model_config.set_path(acquire_instruction)
        # print(new_model_config.path)
        return 0,0
    
    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.Acquistion):
        idx_log = Log.get_config(model_config, acquire_instruction, 'indices')
        Log.save(data, idx_log)

    def get_new_data(self, data_splits: Dataset.DataSplits, new_data_indices, augmentation):
        if augmentation:
            return torch.utils.data.Subset(data_splits.dataset['market_aug'],new_data_indices)
        else:
            return torch.utils.data.Subset(data_splits.dataset['market'],new_data_indices)
        
class Greedy(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    def get_new_data_indices(self, n_data, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel):
        base_model = Model.load(old_model_config)
        clf = CLF.SVM(data_splits.loader['train_clip'])
        score = clf.fit(base_model, data_splits.loader['val_shift'])
        market_info, _ = clf.predict(data_splits.loader['market'])
        new_data_indices_total = []
        for c in range(old_model_config.class_number):
            cls_indices = acquistion.extract_class_indices(c, market_info['gt'])
            cls_dv = market_info['dv'][cls_indices]
            sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
            new_data_indices = acquistion.get_top_values(cls_indices[sorted_idx],n_data)
            new_data_indices_total.append(new_data_indices)
        clf_info = {
            'clf': clf,
            'score': score
        }
        return np.concatenate(new_data_indices_total), clf_info       

class Sample(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    def get_new_data_indices(self, n_data, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel):
        market_gts = acquistion.get_loader_labels(data_splits.loader['market'])
        new_data_indices_total = []
        for c in range(old_model_config.class_number):
            cls_indices = acquistion.extract_class_indices(c, market_gts)  
            new_data_indices = acquistion.sample_acquire(cls_indices,n_data)
            new_data_indices_total.append(new_data_indices)
        clf_info = None
        return np.concatenate(new_data_indices_total), clf_info      

class Confidence(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    def get_new_data_indices(self, n_data, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel):
        base_model = Model.load(old_model_config)
        market_gts, market_preds, market_confs = Model.evaluate(data_splits.loader['market'], base_model)
        new_data_indices_total = []
        for c in range(old_model_config.class_number):   
            cls_indices = acquistion.extract_class_indices(c, market_gts)  
            cls_conf = market_confs[cls_indices]
            conf_sorted_idx = np.argsort(cls_conf) 
            new_data_indices = acquistion.get_top_values(cls_indices[conf_sorted_idx],n_data)
            new_data_indices_total.append(new_data_indices) 
        clf_info = None
        return np.concatenate(new_data_indices_total), clf_info       

class Mix(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    def get_new_data_indices(self, n_data, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel):
        base_model = Model.load(old_model_config)
        clf = CLF.SVM(data_splits.loader['train_clip'])
        score = clf.fit(base_model, data_splits.loader['val_shift'])
        market_info, _ = clf.predict(data_splits.loader['market'])
        new_data_indices_total = []
        for c in range(old_model_config.class_number):
            cls_indices = acquistion.extract_class_indices(c, market_info['gt'])
            cls_dv = market_info['dv'][cls_indices]
            sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
            greedy_results = acquistion.get_top_values(cls_indices[sorted_idx],n_data-n_data//2)
            sample_results = acquistion.sample_acquire(cls_indices,n_data//2)
            new_data_cls_indices = np.concatenate([greedy_results, sample_results])
            new_data_indices_total.append(new_data_cls_indices)
        clf_info = None
        return np.concatenate(new_data_indices_total), clf_info       

class SeqCLF(Strategy):
    def __init__(self) -> None:
        super().__init__()
    def operate(self, acquire_instruction:Config.SequentialAc, dataset:dict, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, old_model_config.batch_size)
        org_val_ds = dataset_splits.dataset['val_shift']
        new_data_total_set = None
        rounds = acquire_instruction.sequential_rounds_info[acquire_instruction.new_data_number_per_class]

        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, dataset_splits, old_model_config)
            new_data_round_set_no_aug = torch.utils.data.Subset(dataset_splits.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(dataset_splits, new_data_round_indices, new_model_config.augment)

            dataset_splits.reduce('market', new_data_round_indices)
            dataset_splits.expand('val_shift', new_data_round_set_no_aug)

            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        dataset_splits.use_new_data(new_data_total_set, new_model_config, acquire_instruction)
        assert len(org_val_ds) == len(dataset_splits.dataset['val_shift']) - acquire_instruction.get_new_data_size(new_model_config.class_number), "size error with original val"
        dataset_splits.replace('val_shift', org_val_ds)
        new_model = Model.get_new(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'])
        new_model_config.set_path(acquire_instruction)
        Model.save(new_model,new_model_config.path)

        clf = clf_info['clf']
        self.log_data(new_model_config, new_data_total_set, acquire_instruction, clf)

    def log_data(self, model_config:Config.NewModel, data, acquire_instruction: Config.SequentialAc, clf):
        data_config = Log.get_config(model_config, acquire_instruction, 'data')
        Log.save(data, data_config) # Save new data
        clf_config = Log.get_config(model_config, acquire_instruction, 'clf')
        Log.save(clf, clf_config) # Save new clf

class Seq(Strategy):
    def __init__(self) -> None:
        super().__init__()
    def operate(self, acquire_instruction:Config.SequentialAc, dataset: dict, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
        dataset_copy = deepcopy(dataset)
        dataset_splits = Dataset.DataSplits(dataset_copy, old_model_config.batch_size)
        new_data_total_set = None
        rounds = acquire_instruction.sequential_rounds_info[acquire_instruction.new_data_number_per_class]
        model = Model.load(old_model_config)
        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, dataset_splits, old_model_config)
            new_data_round_set_no_aug = torch.utils.data.Subset(dataset_splits.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(dataset_splits, new_data_round_indices, new_model_config.augment)

            dataset_splits.reduce('market', new_data_round_indices)
            dataset_splits.expand('train_clip', new_data_round_set_no_aug)
            dataset_splits.expand('train', new_data_round_set)

            model = Model.get_new(new_model_config, dataset_splits.loader['train'], dataset_splits.loader['val_shift'], model)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        assert len(new_data_total_set) == acquire_instruction.get_new_data_size(new_model_config.class_number), 'size error with new data'
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

def StrategyFactory(strategy):
    if strategy=='dv':
        return Greedy()
    elif strategy =='sm':
        return Sample()
    elif strategy == 'conf':
        return Confidence()
    elif strategy == 'mix':
        return Mix()
    elif strategy == 'seq':
        return Seq()
    else:
        return SeqCLF()
