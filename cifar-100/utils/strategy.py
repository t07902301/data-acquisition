from utils.dataset import *
from utils.model import *
from utils.acquistion import *
from utils.Config import *
from copy import deepcopy
from utils.log import *
from abc import abstractmethod
class Strategy():
    old_model_config: OldModelConfig
    new_model_config: NewModelConfig
    def __init__(self, old_model_config, new_model_config) -> None:
        self.old_model_config = old_model_config
        self.new_model_config = new_model_config
    @abstractmethod
    def operate(self, acquire_instruction: AcquistionConfig, data_splits: DataSplits):
        pass

class NonSeqStrategy(Strategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    @abstractmethod
    def get_new_data_indices(self, n_data, data_splits: DataSplits):
        pass
    def operate(self, acquire_instruction: AcquistionConfig, data_splits: DataSplits):
        ds = deepcopy(data_splits)
        ds.get_dataloader(self.new_model_config.batch_size)
        new_data_indices, raw_new_data_indices = self.get_new_data_indices(acquire_instruction.new_data_number_per_class, ds)
        new_data_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_indices)
        ds.use_new_data(new_data_set, self.new_model_config, acquire_instruction)
        new_model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'])
        self.new_model_config.set_path(acquire_instruction)
        save_model(new_model,self.new_model_config.path)
        # log_data(ds.dataset['train'], self.new_model_config, acquire_instruction)
        log_indices(raw_new_data_indices, self.new_model_config, acquire_instruction)
        
class Greedy(NonSeqStrategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    def get_new_data_indices(self, n_data, data_splits: DataSplits):
        base_model = load_model(self.old_model_config)
        clf,clip_processor,_ = get_CLF(base_model,data_splits.loader)
        market_info = apply_CLF(clf,data_splits.loader['market'],clip_processor)
        new_data_indices_total = []
        for c in range(self.new_model_config.class_number):
            cls_indices, cls_mask, cls_dv = get_class_info(c, market_info['gt'], market_info['dv'])
            sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
            new_data_indices = get_top_values(cls_indices[sorted_idx],n_data)
            new_data_indices_total.append(new_data_indices)
            # print(c, np.max(cls_dv[sorted_idx][:n_data]))
        return np.concatenate(new_data_indices_total), new_data_indices_total       

class Sample(NonSeqStrategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    def get_new_data_indices(self, n_data, data_splits: DataSplits):
        market_gts = get_loader_labels(data_splits.loader['market'])
        new_data_indices_total = []
        for c in range(self.new_model_config.class_number):
            cls_indices, cls_mask, _ = get_class_info(c, market_gts)  
            new_data_indices = sample_acquire(cls_indices,n_data)
            new_data_indices_total.append(new_data_indices)
        return np.concatenate(new_data_indices_total), new_data_indices_total       

class Confidence(NonSeqStrategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    def get_new_data_indices(self, n_data, data_splits: DataSplits):
        base_model = load_model(self.old_model_config)
        market_gts, market_preds, market_confs = evaluate_model(data_splits.loader['market'], base_model)
        new_data_indices_total = []
        for c in range(self.new_model_config.class_number):   
            cls_indices, cls_mask, _ = get_class_info(c, market_gts)  
            cls_conf = market_confs[cls_mask]
            conf_sorted_idx = np.argsort(cls_conf) 
            new_data_indices = get_top_values(cls_indices[conf_sorted_idx],n_data)
            new_data_indices_total.append(new_data_indices) 
        return np.concatenate(new_data_indices_total), new_data_indices_total       

class Mix(NonSeqStrategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    def get_new_data_indices(self, n_data, data_splits: DataSplits):
        base_model = load_model(self.old_model_config)
        clf,clip_processor,_ = get_CLF(base_model,data_splits.loader)
        market_info = apply_CLF(clf,data_splits.loader['market'],clip_processor)
        new_data_indices_total = []
        for c in range(self.new_model_config.class_number):
            cls_indices, cls_mask, cls_dv = get_class_info(c, market_info['gt'], market_info['dv'])
            sorted_idx = np.argsort(cls_dv) # index of images ordered by their decision values
            greedy_results = get_top_values(cls_indices[sorted_idx],n_data-n_data//2)
            sample_results = sample_acquire(cls_indices,n_data//2)
            new_data_indices = np.concatenate([greedy_results, sample_results])
            new_data_indices_total.append(new_data_indices)
        return np.concatenate(new_data_indices_total), new_data_indices_total       

class SeqCLF(Strategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)
    def operate(self, acquire_instruction:SequentialAcConfig, data_splits: DataSplits):
        self.sub_strategy = StrategyFactory(self.old_model_config, self.new_model_config, acquire_instruction.round_acquire_method)
        ds = deepcopy(data_splits)
        ds.get_dataloader(self.new_model_config.batch_size)
        org_val_ds = ds.dataset['val']
        minority_cnt = 0
        new_data_total_set = None
        # rounds = acquire_instruction.sequential_rounds + 1 #the first clf is for data selection
        rounds = acquire_instruction.sequential_rounds

        for round_i in range(rounds):

            acquire_instruction.set_round(round_i)

            assert len(ds.dataset['market'])==len(ds.dataset['market_aug']), "market set size not equal to aug market in round {}".format(round_i)

            new_data_round_indices,_ = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, ds)
            new_data_round_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_round_indices)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)

            ds.reduce('market', new_data_round_indices, self.new_model_config.batch_size, acquire_instruction)
            ds.reduce('market_aug', new_data_round_indices, self.new_model_config.batch_size, acquire_instruction)
            ds.expand('val', new_data_round_set_no_aug, self.new_model_config.batch_size, acquire_instruction)

            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        ds.use_new_data(new_data_total_set, self.new_model_config, acquire_instruction)

        log_data(ds.dataset['train'], self.new_model_config, acquire_instruction)
    
        assert len(org_val_ds) == len(ds.dataset['val']) - acquire_instruction.get_new_data_size(self.new_model_config.class_number), "size error with original val"

        ds.update_dataset('val', org_val_ds, self.new_model_config.batch_size)

        new_model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'])
        self.new_model_config.set_path(acquire_instruction)
        save_model(new_model,self.new_model_config.path)

class Seq(Strategy):
    def __init__(self, old_model_config, new_model_config) -> None:
        super().__init__(old_model_config, new_model_config)   
    def operate(self, acquire_instruction:SequentialAcConfig, data_splits: DataSplits):
        self.sub_strategy = StrategyFactory(self.old_model_config, self.new_model_config, acquire_instruction.round_acquire_method)
        ds = deepcopy(data_splits)
        ds.get_dataloader(self.new_model_config.batch_size)
        pure = self.new_model_config.pure
        minority_cnt = 0
        new_data_total_set = None
        rounds = acquire_instruction.sequential_rounds

        model = load_model(self.old_model_config)

        for round_i in range(rounds):
            acquire_instruction.set_round(round_i)
            new_data_round_indices,_ = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, ds)
            new_data_round_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_round_indices)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)

            ds.reduce('market', new_data_round_indices, self.new_model_config.batch_size, acquire_instruction)
            ds.reduce('market_aug', new_data_round_indices, self.new_model_config.batch_size, acquire_instruction)
            ds.expand('train', new_data_round_set, self.new_model_config.batch_size, acquire_instruction)
            ds.expand('train_clip', new_data_round_set_no_aug, self.new_model_config.batch_size, acquire_instruction)

            model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'], model)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

        assert len(new_data_total_set) == acquire_instruction.get_new_data_size(self.new_model_config.class_number), 'size error with new data'
        if pure:
            ds.update_dataset('train', new_data_total_set, self.new_model_config.batch_size)
            model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'], model)

        log_data(ds.dataset['train'], self.new_model_config, acquire_instruction)
        self.new_model_config.set_path(acquire_instruction)
        save_model(model,self.new_model_config.path)

def StrategyFactory(old_model_config, new_model_config,strategy):
    if strategy=='dv':
        return Greedy(old_model_config, new_model_config)
    elif strategy =='sm':
        return Sample(old_model_config, new_model_config)
    elif strategy == 'conf':
        return Confidence(old_model_config, new_model_config)
    elif strategy == 'mix':
        return Mix(old_model_config, new_model_config)
    elif strategy == 'seq':
        return Seq(old_model_config, new_model_config)
    else:
        return SeqCLF(old_model_config, new_model_config)
