from copy import deepcopy
from abc import abstractmethod
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
from utils.objects.log import Log
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
import torch
import numpy as np
import utils.statistics.data as DataStat
import utils.statistics.partitioner as Partitioner
import utils.statistics.distribution as Distribution
from utils.env import data_split_env

#TODO add info class with base model and dataset, make strategy class more purified. 

class WorkSpace():
    '''
    Data + Base Model: reset before each strategy operation
    '''
    def __init__(self, model_config: Config.OldModel, dataset:dict) -> None:
        self.base_model_config = model_config
        self.init_dataset = dataset
    
    def set_up(self, new_batch_size, clip_processor):
        '''
        set up base model + datasplits
        '''
        self.base_model = Model.prototype_factory(self.base_model_config.base_type, self.base_model_config.class_number, clip_processor=clip_processor)
        self.base_model.load(self.base_model_config.path, self.base_model_config.device)
        copied_dataset = deepcopy(self.init_dataset)
        self.data_split = Dataset.DataSplits(copied_dataset, new_batch_size)

    def set_validation(self, stream_instruction:Config.ProbabStream, old_batch_size, new_batch_size, clf: Detector.Prototype=None, detect_instruction:Config.Detection=None):
        '''
        align val_shift with test splits\n
        NonSeq has CLF fixed and get it again\n
        Seq has CLF updated and restore it from logs
        '''

        if clf is None and detect_instruction != None:
            # TODO unify detector args
            clf = Detector.factory(detect_instruction.name, clip_processor = detect_instruction.vit, split_and_search=True)
            _ = clf.fit(self.base_model, self.data_split.loader['val_shift']) 

        correct_dstr = Distribution.get_correctness_dstr(self.base_model, clf, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=True)
        incorrect_dstr = Distribution.get_correctness_dstr(self.base_model, clf, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=False)

        val_shift_info = DataStat.build_info(self.data_split, 'val_shift', clf, old_batch_size, new_batch_size, self.base_model)
        val_shift_split, _ = Partitioner.Probability().run(val_shift_info, {'target': incorrect_dstr, 'other': correct_dstr}, stream_instruction)
        self.validation_loader = val_shift_split['new_model']

    def reset(self, new_batch_size, clip_processor):
        self.set_up(new_batch_size, clip_processor)
        
class Strategy():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        '''
        Under an acquistion instruction (which method, how much new data), this strategy operates on data and the old Model. Finally, the new model will be saved. \n
        Init workspace should be identical across every method and n_data
        '''
        pass

    @abstractmethod
    def get_new_data_indices(self, operation:Config.Operation, workspace:WorkSpace):
        '''
        Based on old model performance, a certain number of new data is obtained.
        '''
        pass

    def get_new_data_info(self, operation: Config.Operation, workspace:WorkSpace):

        new_data_indices, detector = self.get_new_data_indices(operation, workspace)
        return {
            'data': torch.utils.data.Subset(workspace.data_split.dataset['market'], new_data_indices),
            'indices': new_data_indices,
            'clf': detector
        }
    
    def test_new_data(self):
        pass
        # set_indices = []
        # for img in range(len(new_data)):
        #     idx = new_data[img][3]
        #     set_indices.append(idx)
        # print('New data real indices:', set_indices)

    def test_clf(self):
        pass

class NonSeqStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, n_data, market_loader):
        pass

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)

        new_data_info = self.get_new_data_info(operation, workspace)
        workspace.data_split.use_new_data(new_data_info['data'], new_model_config, operation.acquisition)

        new_model_config.set_path(operation)

        workspace.base_model.update(new_model_config, workspace.data_split.loader['train'], workspace.validation_loader)

        workspace.base_model.save(new_model_config.path)

        self.export_indices(new_model_config, operation.acquisition, new_data_info['indices'], operation.stream)

    def export_indices(self, model_config:Config.NewModel, acquire_instruction: Config.Acquisition, data, stream: Config.Stream):
        if model_config.check_rs(acquire_instruction.method, stream.bound) is False:
            log = Log(model_config, 'indices')
            log.export(acquire_instruction, data=data)

class Greedy(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()

    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_n_data = operation.acquisition.n_ndata
        detector_instruction = operation.detection
        detector_train_dict = {'loader': dataset_splits.loader['val_shift'], 'dataset': dataset_splits.dataset['val_shift']}
        detector = self.get_Detector(detector_instruction, workspacce.base_model, detector_train_dict)
        new_data_indices = self.run(acquistion_n_data, dataset_splits.loader['market'], detector, workspacce.base_model)
        return new_data_indices, detector  

    def get_Detector(self, detector_instruction: Config.Detection, base_model: Model.prototype, validation_dict):
        val_loader, val_dataset = validation_dict['loader'], validation_dict['dataset']
        detector = Detector.factory(detector_instruction.name, clip_processor = detector_instruction.vit)
        _ = detector.fit(base_model, val_loader, val_dataset, batch_size=None)
        return detector
    
    def run(self, n_data, market_loader, detector: Detector.Prototype, base_model:Model.prototype, bound=None):
        market_dv, _ = detector.predict(market_loader, base_model)
        if bound == None:
            new_data_indices = acquistion.get_top_values_indices(market_dv, n_data)
        else:
            new_data_indices = acquistion.get_in_bound_top_indices(market_dv, n_data, bound)
        return new_data_indices

class Sample(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
        data_split_env()

    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_n_data = operation.acquisition.n_ndata
        new_data_indices = self.run(acquistion_n_data, dataset_splits.loader['market'])
        clf_info = None
        return new_data_indices, clf_info      
    
    def run(self, n_data, market_loader):
        market_gts = acquistion.get_loader_labels(market_loader)
        new_data_indices = acquistion.sample_acquire(market_gts, n_data)
        return new_data_indices

class Confidence(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    
    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_n_data = operation.acquisition.n_ndata
        model_n_class = workspacce.base_model_config.class_number
        new_data_indices = self.run(acquistion_n_data, dataset_splits.loader['market'], workspacce.base_model, model_n_class)
        clf_info = None
        return new_data_indices, clf_info      

    def run(self, n_data, market_loader, base_model:Model.prototype, model_n_class):
        market_gts, _, market_probab = base_model.eval(market_loader)
        if model_n_class == 1:
            confs = acquistion.get_probab_diff(market_gts, market_probab)
        else:
            confs = acquistion.get_probab_gts(market_gts, market_probab)
        new_data_indices = acquistion.get_top_values_indices(confs, n_data)
        return new_data_indices

# class Mix(NonSeqStrategy):
#     def __init__(self) -> None:
#         super().__init__()
#         Dataset.data_split_env()

#     def get_new_data_indices(self, n_data, dataset_splits: Dataset.DataSplits, detector_instruction: Config.Detection, bound = None):
#         clf = Detector.SVM(dataset_splits.loader['train_clip'], self.clip_processor)
#         _ = clf.fit(self.base_model, dataset_splits.loader['val_shift'])
#         market_dv, _ = clf.predict(dataset_splits.loader['market'], self.base_model)
#         greedy_results = acquistion.get_top_values_indices(market_dv, n_data-n_data//2)
#         sample_results = acquistion.sample_acquire(market_dv,n_data//2)
#         new_data_cls_indices = np.concatenate([greedy_results, sample_results])
#         clf_info = None
#         return new_data_cls_indices, clf_info       

class SeqCLF(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)
        operation.acquisition.set_up()
        self.sub_strategy = StrategyFactory(operation.acquisition.round_acquire_method)
        new_data_total_set = None
        
        for round_i in range(operation.acquisition.n_rounds):
            new_data_round_info = self.round_operate(round_i, operation, workspace)
            new_data_total_set = new_data_round_info['data'] if round_i==0 else torch.utils.data.ConcatDataset([new_data_total_set, new_data_round_info['data']])

        new_model_config.set_path(operation)
        last_clf = new_data_round_info['clf']

        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)
        workspace.set_validation(operation.stream, new_model_config.batch_size, new_model_config.new_batch_size, last_clf)

        # train model 
        workspace.data_split.use_new_data(new_data_total_set, new_model_config, operation.acquisition)

        workspace.base_model.update(new_model_config.setter, workspace.data_split.loader['train'], workspace.validation_loader)

        workspace.base_model.save(new_model_config.path)

        self.export_clf(new_model_config, operation.acquisition, last_clf)
        self.export_data(new_model_config, operation.acquisition, new_data_total_set)

    def round_operate(self, round_id, operation: Config.Operation, workspace:WorkSpace):
        sub_operation = self.round_set_up(operation, round_id)    
        data_split = workspace.data_split

        new_data_round_info = self.sub_strategy.get_new_data_info(sub_operation, workspace)

        data_split.reduce('market', new_data_round_info['indices'])
        data_split.expand('val_shift', new_data_round_info['data'])

        return new_data_round_info

    def round_set_up(self, operation: Config.Operation, round_id):
        '''
        Build sub operation where n_ndata is replaced by n_data_round
        '''
        operation.acquisition.set_round(round_id)
        sub_operation = deepcopy(operation)
        sub_operation.acquisition.n_ndata = operation.acquisition.n_data_round    
        return sub_operation

    def export_clf(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, detector: Detector.Prototype):
        log = Log(model_config, 'clf')
        log.export(acquire_instruction, detector=detector)

    def export_data(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, dataset):
        log = Log(model_config, 'data')
        log.export(acquire_instruction, data=dataset)
        
    def get_new_data_indices(self, operation:Config.Operation, workspace:WorkSpace):
        pass
# class Seq(Strategy):
#     def __init__(self) -> None:
#         super().__init__()
#     def operate(self, acquire_instruction:Config.SequentialAc, dataset: dict, old_model_config: Config.OldModel, new_model_config:Config.NewModel):
#         self.sub_strategy = StrategyFactory(acquire_instruction.round_acquire_method)
#         dataset_copy = deepcopy(dataset)
#         workspace = Dataset.DataSplits(dataset_copy, old_model_config.batch_size)
#         new_data_total_set = None
#         rounds = 2
#         model = Model.load(workspace)
#         for round_i in range(rounds):
#             acquire_instruction.set_round(round_i)
#             new_data_round_indices, clf_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, workspace, old_model_config)
#             new_data_round_set = torch.utils.data.Subset(workspace.inidataset['market'],new_data_round_indices)
#             new_data_round_set = self.sub_strategy.get_new_data(workspace, new_data_round_indices, )

#             workspace.reduce('market', new_data_round_indices)
#             workspace.expand('train_clip', new_data_round_set)
#             workspace.expand('train', new_data_round_set)

#             model = Model.get_new(new_model_config, workspace.data_split.loader['train'], workspace.data_split.loader['val_shift'], model)
#             new_data_total_set = new_data_round_set  if (new_data_total_set == None) else torch.utils.data.ConcatDataset([new_data_total_set,new_data_round_set])

#         assert len(new_data_total_set) == acquire_instruction.n_ndata, 'size error with new data'
#         if new_model_config.pure:
#             workspace.replace('train', new_data_total_set)
#             model = Model.get_new(new_model_config, workspace.data_split.loader['train'], workspace.data_split.loader['val_shift'], model)

#         new_model_config.set_path(acquire_instruction)
#         Model.save(model,new_model_config.path)

#         clf = clf_info['clf']
#         self.export_log(new_model_config, new_data_total_set, acquire_instruction, clf)

#     def export_log(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, detector: Detector.Prototype):
#         log = Log(model_config, 'clf')
#         log.export(acquire_instruction, detector=detector)

def StrategyFactory(strategy):
    if strategy=='dv':
        return Greedy()
    elif strategy =='sm':
        return Sample()
    elif strategy == 'conf':
        return Confidence()
    # elif strategy == 'mix':
    #     return Mix()
    # elif strategy == 'seq':
    #     return Seq()
    else:
        return SeqCLF()