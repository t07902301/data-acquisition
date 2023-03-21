from utils.dataset import *
from utils.model import *
from utils.acquistion import *
from utils.Config import *
from copy import deepcopy
class Strategy():
    data: DataSplits
    old_model_config: OldModelConfig
    new_model_config: NewModelConfig
    def __init__(self, data, old_model_config, new_model_config) -> None:
        self.data = data
        self.old_model_config = old_model_config
        self.new_model_config = new_model_config
class NonSeqStrategy(Strategy):
    def __init__(self, data, old_model_config, new_model_config) -> None:
        super().__init__(data, old_model_config, new_model_config)
    def get_new_model(self, acquisition_config:AcquistionConfig):
        ds = deepcopy(self.data)
        ds.get_dataloader(self.new_model_config.batch_size)

        base_model = load_model(self.old_model_config.path)
        # Get SVM 
        clf,clip_processor,_ = get_CLF(base_model,ds.loader)
        market_info = apply_CLF(clf,ds.loader['market'],clip_processor,base_model)

        new_data_indices = get_new_data_indices(method=acquisition_config.method, new_data_number_per_class=acquisition_config.new_data_number_per_class, class_number=self.new_model_config.class_number,ds_info=market_info)
        new_data_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_indices)

        ds.use_new_data(new_data_set, self.new_model_config, acquisition_config)
        log_data(ds.dataset['train'], self.new_model_config, acquisition_config)

        new_model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'])
        return new_model
class SeqStrategy(Strategy):
    def __init__(self, data, old_model_config, new_model_config) -> None:
        super().__init__(data, old_model_config, new_model_config)
    def seq_clf(self, acquisition_config:SequentialAcConfig):
        ds = deepcopy(self.data)
        ds.get_dataloader(self.new_model_config.batch_size)
        org_val_ds = ds.dataset['val']
        minority_cnt = 0
        new_data_total_set = None
        rounds = acquisition_config.sequential_rounds
        model = load_model(self.old_model_config.path)
        clf,clip_processor,_ = get_CLF(model,ds.loader)

        for round_i in range(rounds):

            acquisition_config.set_round(round_i)

            assert len(ds.dataset['market'])==len(ds.dataset['market_aug']), "market set size not equal to aug market in round {}".format(round_i)
            market_info = apply_CLF(clf,ds.loader['market'],clip_processor,model)

            new_data_round_indices = get_new_data_indices(method=acquisition_config.round_acquire_method, new_data_number_per_class=acquisition_config.round_data_per_class, class_number=self.new_model_config.class_number, ds_info=market_info)
            new_data_round_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_round_indices)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)

            ds.reduce('market', new_data_round_indices, self.new_model_config.batch_size, acquisition_config)
            ds.reduce('market_aug', new_data_round_indices, self.new_model_config.batch_size, acquisition_config)
            ds.expand('val', new_data_round_set_no_aug, self.new_model_config.batch_size, acquisition_config)

            # update SVM
            clf,clip_processor,_ = get_CLF(model,ds.loader)
            minority_cnt += count_minority(new_data_round_set_no_aug)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else  torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])
        
        ds.use_new_data(new_data_total_set, self.new_model_config, acquisition_config)

        log_data(ds.dataset['train'], self.new_model_config, acquisition_config)
    
        assert len(org_val_ds) == len(ds.dataset['val']) - acquisition_config.get_new_data_size(self.new_model_config.class_number), "size error with original val"

        ds.update_dataset('val', org_val_ds, self.new_model_config.batch_size)

        new_model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'])
        return new_model, minority_cnt

    def seq(self, acquisition_config:SequentialAcConfig):
        ds = deepcopy(self.data)
        ds.get_dataloader(self.new_model_config.batch_size)
        pure = self.new_model_config.pure
        minority_cnt = 0
        new_data_total_set = None
        rounds = acquisition_config.sequential_rounds

        model = load_model(self.old_model_config.path)

        for round_i in range(rounds):

            acquisition_config.set_round(round_i)

            # Get SVM 
            clf,clip_processor,_ = get_CLF(model,ds.loader) #CLF: a metric of model and data

            assert len(ds.dataset['market'])==len(ds.dataset['market_aug']), "market set size not equal to aug market in round {}".format(acquisition_config.current_round)
            market_info = apply_CLF(clf,ds.loader['market'],clip_processor,model)

            new_data_round_indices = get_new_data_indices(method=acquisition_config.round_acquire_method, new_data_number_per_class=acquisition_config.round_data_per_class,class_number=self.new_model_config.class_number,ds_info=market_info)
            new_data_round_set = torch.utils.data.Subset(ds.dataset['market_aug'],new_data_round_indices)
            new_data_round_set_no_aug = torch.utils.data.Subset(ds.dataset['market'],new_data_round_indices)

            ds.reduce('market', new_data_round_indices, self.new_model_config.batch_size, acquisition_config)
            ds.reduce('market_aug', new_data_round_indices, self.new_model_config.batch_size, acquisition_config)
            ds.expand('train', new_data_round_set, self.new_model_config.batch_size, acquisition_config)
            ds.expand('train_clip', new_data_round_set_no_aug, self.new_model_config.batch_size, acquisition_config)

            model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'], model)
            
            minority_cnt += count_minority(new_data_round_set)
            new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])
       
        assert len(new_data_total_set) == acquisition_config.get_new_data_size(self.new_model_config.class_number), 'size error with new data'
        if pure:
            ds.update_dataset('train', new_data_total_set, self.new_model_config.batch_size)
            model = get_new_model(self.new_model_config, ds.loader['train'], ds.loader['val'], model)

        log_data(ds.dataset['train'], self.new_model_config, acquisition_config)

        return model, minority_cnt   

    def get_new_model(self, acquisition_config:SequentialAcConfig):
        if acquisition_config.method=='seq':
            return self.seq(acquisition_config)
        else:
            return self.seq_clf(acquisition_config)

