from copy import deepcopy
from abc import abstractmethod

import utils.log as log
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
    def operate(self, acquire_instruction: Config.Acquistion, data_splits: Dataset.DataSplits, old_model_config:Config.OldModel, new_model_config:Config.NewModel):
        '''
        Under some acquistion instruction (which method, how much new data), this strategy operates on data and the old Model. Finally, the new model will be saved.
        '''
        pass

class NonSeqStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
    @abstractmethod
    def get_new_data_indices(self, n_data, data_splits: Dataset.DataSplits, old_model_config:Config.OldModel):
        '''
        Based on old model performance, a certain number of new data is acquired.
        '''
        pass
    def operate(self, acquire_instruction: Config.Acquistion, data_splits: Dataset.DataSplits, old_model_config:Config.OldModel, new_model_config:Config.NewModel):
        ds = deepcopy(data_splits)
        ds.get_dataloader(new_model_config.batch_size)
        new_data_indices, raw_new_data_indices,_ = self.get_new_data_indices(acquire_instruction.new_data_number_per_class, ds, old_model_config)
        # new_data_set = self.get_new_data(data_splits, new_data_indices, new_model_config.augment)
        # ds.use_new_data(new_data_set, new_model_config, acquire_instruction)
        # new_model = Model.get_new(new_model_config, ds.loader['train'], ds.loader['val_shift'])
        # new_model_config.set_path(acquire_instruction)
        # Model.save(new_model,new_model_config.path)
        if new_model_config.pure:
            idx_log = log.get_sub_log('indices',new_model_config, acquire_instruction)
            log.save(raw_new_data_indices, idx_log)
            base_model = Model.load(old_model_config)
            clf = CLF.SVM(ds.loader['train_clip'])
            score = clf.fit(base_model, ds.loader['val_shift'])
            market_info, precision = clf.predict(ds.loader['market'])
            for cls_idx in raw_new_data_indices:
                print(np.max(market_info['dv'][cls_idx]))
            new_model_config.set_path(acquire_instruction)
            print(new_model_config.path)

        return 0,0

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
        return np.concatenate(new_data_indices_total), new_data_indices_total, clf_info       

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
        return np.concatenate(new_data_indices_total), new_data_indices_total, clf_info      

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
        return np.concatenate(new_data_indices_total), new_data_indices_total, clf_info       

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
        return np.concatenate(new_data_indices_total), new_data_indices_total, clf_info       

class SeqCLF(Strategy):
    def __init__(self) -> None:
        super().__init__()
    def operate(self, acquire_instruction:Config.SequentialAc, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
        ds = deepcopy(data_splits)
        ds.get_dataloader(new_model_config.batch_size)
        org_val_ds = ds.dataset['val_shift']
        new_data_total_set = None
        rounds = acquire_instruction.sequential_rounds_info[acquire_instruction.new_data_number_per_class]

        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, _, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, ds, old_model_config)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(ds, new_data_round_indices, new_model_config.augment)

            ds.reduce('market', new_data_round_indices, new_model_config.batch_size)
            ds.expand('val_shift', new_data_round_set_no_aug, new_model_config.batch_size)

            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        ds.use_new_data(new_data_total_set, new_model_config, acquire_instruction)
        assert len(org_val_ds) == len(ds.dataset['val_shift']) - acquire_instruction.get_new_data_size(new_model_config.class_number), "size error with original val"
        ds.update_dataset('val_shift', org_val_ds, new_model_config.batch_size)
        new_model = Model.get_new(new_model_config, ds.loader['train'], ds.loader['val_shift'])
        new_model_config.set_path(acquire_instruction)
        Model.save(new_model,new_model_config.path)

        if new_model_config.pure:
            base_model = Model.load(old_model_config)
            gt, pred,_  = Model.evaluate(ds.loader['test'],base_model)
            _, precision = clf_info['clf'].predict(ds.loader['test'], compute_metrics=True, dataset_preds= pred)        
            stat_config = log.get_sub_log('stat', new_model_config, acquire_instruction)
            stat = {
                'cv score': clf_info['score'],
                'precision': precision
            }
            log.save(stat, stat_config)

            data_config = log.get_sub_log('data', new_model_config, acquire_instruction)
            log.save(new_data_total_set, data_config) # Save new data

            # clf_config = log.get_sub_log('clf', new_model_config, acquire_instruction)
            # clf_data = {
            #     'train_clip': ds.dataset['train_clip'],
            #     'val_shift': ds.dataset['val_shift']
            # } 
            # log.save(clf_data, clf_config)

class Seq(Strategy):
    def __init__(self) -> None:
        super().__init__()
    def operate(self, acquire_instruction:Config.SequentialAc, data_splits: Dataset.DataSplits, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
        self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
        ds = deepcopy(data_splits)
        ds.get_dataloader(new_model_config.batch_size)
        new_data_total_set = None
        rounds = acquire_instruction.sequential_rounds_info[acquire_instruction.new_data_number_per_class]
        model = Model.load(old_model_config)
        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices, _, _ = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, ds, old_model_config)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)
            new_data_round_set = self.sub_strategy.get_new_data(ds, new_data_round_indices, new_model_config.augment)

            ds.reduce('market', new_data_round_indices, new_model_config.batch_size)
            ds.expand('train_clip', new_data_round_set_no_aug, new_model_config.batch_size)
            ds.expand('train', new_data_round_set, new_model_config.batch_size)

            model = Model.get_new(new_model_config, ds.loader['train'], ds.loader['val_shift'], model)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        assert len(new_data_total_set) == acquire_instruction.get_new_data_size(new_model_config.class_number), 'size error with new data'
        if new_model_config.pure:
            ds.update_dataset('train', new_data_total_set, new_model_config.batch_size)
            model = Model.get_new(new_model_config, ds.loader['train'], ds.loader['val_shift'], model)

        # log_data(ds.dataset['train'], new_model_config, acquire_instruction)
        new_model_config.set_path(acquire_instruction)
        Model.save(model,new_model_config.path)

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
