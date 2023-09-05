from copy import deepcopy
from abc import abstractmethod
import utils.acquistion as acquistion
import utils.objects.model as Model
import utils.objects.Config as Config
from utils.objects.log import Log
import utils.dataset.wrappers as dataset_utils
import utils.objects.Detector as Detector
import torch
import numpy as np
import utils.statistics.data as DataStat
import utils.statistics.partitioner as Partitioner
import utils.statistics.distribution as distribution_utils
from utils.env import data_split_env
import utils.ood as OOD
import utils.objects.dataloader as dataloader_utils
from utils.logging import logger

#TODO add info class with base model and dataset, make strategy class more purified. 

class WorkSpace():
    '''
    Data + Base Model: reset before each strategy operation
    '''
    def __init__(self, model_config: Config.OldModel, dataset:dict, config) -> None:
        self.base_model_config = model_config
        self.init_dataset = dataset
        self.general_config = config
        self.market_dataset = None
        self.validation_loader = None
        self.detector = None
        self.base_model = None
        self.data_split = None
    
    def set_up(self, new_batch_size, clip_processor):
        '''
        set up base model + datasplits
        '''
        self.set_model(clip_processor)
        self.set_data(new_batch_size)

    def set_model(self, clip_processor):
        del self.base_model
        self.base_model = Model.factory(self.base_model_config.base_type, self.general_config, clip_processor=clip_processor)
        self.base_model.load(self.base_model_config.path, self.base_model_config.device)

    def set_data(self, new_batch_size):
        del self.data_split
        copied_dataset = deepcopy(self.init_dataset)
        self.data_split = dataset_utils.DataSplits(copied_dataset, new_batch_size)

    def set_validation(self, stream_instruction:Config.ProbabStream, old_batch_size, new_batch_size):
        '''
        align val_shift with test splits\n
        Error detector from workspace (fixed or from the last round in seq)
        '''
        correct_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=False)

        val_shift_info = DataStat.build_info(self.data_split, 'val_shift', self.detector, old_batch_size, new_batch_size)
        val_shift_split, _ = Partitioner.Probability().run(val_shift_info, {'target': incorrect_dstr, 'other': correct_dstr}, stream_instruction)
        self.validation_loader = val_shift_split['new_model']

    def reset(self, new_batch_size, clip_processor):
        self.set_up(new_batch_size, clip_processor)
        if self.market_dataset != None:
            self.data_split.replace('market', self.market_dataset) # For OOD
            logger.info('After filtering, Market size:{}'.format(len(self.data_split.dataset['market'])))

    def set_market(self, clip_processor, known_labels):
        del self.market_dataset

        cover_market_dataset = OOD.run(self.data_split, clip_processor, known_labels)
        self.market_dataset = cover_market_dataset

    def set_detector(self, detector_instruction: Config.Detection):
        val_loader, val_dataset = self.data_split.loader['val_shift'], self.data_split.dataset['val_shift']
        detector = Detector.factory(detector_instruction.name, self.general_config, detector_instruction.vit)
        detector.fit(self.base_model, val_loader, val_dataset, batch_size=None)
        self.detector = detector

    def reset_detector(self, detector_instruction: Config.Detection):
        self.set_detector(detector_instruction)
    
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

        new_data_indices = self.get_new_data_indices(operation, workspace)

        if workspace.base_model_config.base_type == 'cnn':

            new_data = torch.utils.data.Subset(workspace.data_split.dataset['aug_market'], new_data_indices)
        
        else:
            new_data = torch.utils.data.Subset(workspace.data_split.dataset['market'], new_data_indices)
        
        new_data_info = {
            'data': new_data,
            'indices': new_data_indices,
        }

        return new_data_info          
    
    def test_new_data(self):
        pass
        # set_indices = []
        # for img in range(len(new_data)):
        #     idx = new_data[img][3]
        #     set_indices.append(idx)
        # logger.info('New data real indices:', set_indices)

    def test_detector(self):
        pass

    def update_model(self, new_model_config: Config.NewModel, workspace: WorkSpace):
        '''
        Update model after training set in worksapce is refreshed
        '''
        if new_model_config.base_type == 'cnn':
            workspace.base_model.update(new_model_config, workspace.data_split.loader['train'], workspace.validation_loader)
        else:
            workspace.base_model.update(new_model_config, workspace.data_split.loader['train_non_cnn'])

    def update_dataset(self, new_model_config: Config.NewModel, workspace: WorkSpace, acquisition:Config.Acquisition, new_data):
        if new_model_config.base_type == 'cnn':
            workspace.data_split.use_new_data(new_data, new_model_config, acquisition, target_name='train')
        else:
            workspace.data_split.use_new_data(new_data, new_model_config, acquisition, target_name='train_non_cnn')

class NonSeqStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, n_data, market_loader):
        pass

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)

        new_data_info = self.get_new_data_info(operation, workspace)

        self.update_dataset(new_model_config, workspace, operation.acquisition, new_data_info['data'])

        new_model_config.set_path(operation)

        self.update_model(new_model_config, workspace)

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
        new_data_indices = self.run(acquistion_n_data, dataset_splits.loader['market'], workspacce.detector)
        return new_data_indices  
    
    def run(self, n_data, market_loader, detector: Detector.Prototype, bound=None):
        market_dv, _ = detector.predict(market_loader)
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
        new_data_indices = self.run(acquistion_n_data, dataset_splits.dataset['market'])
        return new_data_indices      
    
    def run(self, n_data, market_dataset):
        # Market sampling does not know data subclass labels which are often inaccessible
        market_indices = np.arange(len(market_dataset))
        new_data_indices = acquistion.sample(market_indices, n_data)
        return new_data_indices

class Confidence(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    
    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_n_data = operation.acquisition.n_ndata
        new_data_indices = self.run(acquistion_n_data, dataset_splits.loader['market'], workspacce.base_model, workspacce.base_model_config)
        return new_data_indices      

    def get_confs_score(self,  base_model_config: Config.OldModel, base_model:Model.Prototype,market_loader):

        market_gts, _, market_score = base_model.eval(market_loader)

        if base_model_config.class_number == 1:
            if base_model_config.base_type == 'cnn':
                confs = acquistion.get_probab_diff(market_gts, market_score)
            else:
                confs = acquistion.get_distance_diff(market_gts, market_score)
        else:
            confs = acquistion.get_probab(market_gts, market_score)
        return confs

    def run(self, n_data, market_loader,  base_model:Model.Prototype, base_model_config: Config.OldModel):

        confs = self.get_confs_score(base_model_config, base_model, market_loader)

        new_data_indices = acquistion.get_top_values_indices(confs, n_data)

        return new_data_indices

# class Mix(NonSeqStrategy):
#     def __init__(self) -> None:
#         super().__init__()
#         dataset_utils.data_split_env()

#     def get_new_data_indices(self, n_data, dataset_splits: dataset_utils.DataSplits, detector_instruction: Config.Detection, bound = None):
#         detector = Detector.SVM(dataset_splits.loader['train_clip'], self.clip_processor)
#         _ = detector.fit(self.base_model, dataset_splits.loader['val_shift'])
#         market_dv, _ = detector.predict(dataset_splits.loader['market'], self.base_model)
#         greedy_results = acquistion.get_top_values_indices(market_dv, n_data-n_data//2)
#         sample_results = acquistion.sample_acquire(market_dv,n_data//2)
#         new_data_cls_indices = np.concatenate([greedy_results, sample_results])
#         detector_info = None
#         return new_data_cls_indices, detector_info       

class SeqCLF(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)
        workspace.reset_detector(operation.detection)
        operation.acquisition.set_up()

        self.sub_strategy = StrategyFactory(operation.acquisition.round_acquire_method)
        new_data_total_set = None
        for round_i in range(operation.acquisition.n_rounds):
            new_data_round_info = self.round_operate(round_i, operation, workspace)
            new_data_total_set = new_data_round_info['data'] if round_i==0 else torch.utils.data.ConcatDataset([new_data_total_set, new_data_round_info['data']])

        new_model_config.set_path(operation)
        logger.info(new_model_config.path)

        workspace.set_data(new_model_config.new_batch_size) # recover validation
        workspace.set_validation(operation.stream, new_model_config.batch_size, new_model_config.new_batch_size)

        self.update_dataset(new_model_config, workspace, operation.acquisition, new_data_total_set)

        self.update_model(new_model_config, workspace)

        workspace.base_model.save(new_model_config.path)

        self.export_detector(new_model_config, operation.acquisition, workspace.detector)
        self.export_data(new_model_config, operation.acquisition, new_data_total_set)

    def round_operate(self, round_id, operation: Config.Operation, workspace:WorkSpace):
        '''
        Get new data and update error detector in the workspace
        '''
        round_operation = self.round_set_up(operation, round_id)    
        data_split = workspace.data_split

        new_data_round_info = self.sub_strategy.get_new_data_info(round_operation, workspace)

        data_split.reduce('market', new_data_round_info['indices'])
        data_split.reduce('aug_market', new_data_round_info['indices'])
        data_split.expand('val_shift', new_data_round_info['data'])

        workspace.set_detector(operation.detection)

        return new_data_round_info

    def round_set_up(self, operation: Config.Operation, round_id):
        '''
        Set up data in each round (by round ID)
        '''
        round_operation = deepcopy(operation)
        if operation.acquisition.seq_mode == 'n_data':
            round_operation.acquisition.n_ndata = operation.acquisition.n_ndata_round
            return round_operation
        else:
            round_operation.acquisition.current_round = round_id + 1
            if round_operation.acquisition.current_round == round_operation.acquisition.n_rounds:
                round_operation.acquisition.n_ndata = operation.acquisition.n_data_last_round
            else:
                round_operation.acquisition.n_ndata = operation.acquisition.n_data_non_last_round
            return round_operation

    def export_detector(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, detector: Detector.Prototype):
        log = Log(model_config, 'detector')
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
#         workspace = dataset_utils.DataSplits(dataset_copy, old_model_config.batch_size)
#         new_data_total_set = None
#         rounds = 2
#         model = Model.load(workspace)
#         for round_i in range(rounds):
#             acquire_instruction.set_round(round_i)
#             new_data_round_indices, detector_info = self.sub_strategy.get_new_data_indices(acquire_instruction.round_data_per_class, workspace, old_model_config)
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

#         detector = detector_info['detector']
#         self.export_log(new_model_config, new_data_total_set, acquire_instruction, detector)

#     def export_log(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, detector: Detector.Prototype):
#         log = Log(model_config, 'detector')
#         log.export(acquire_instruction, detector=detector)

def StrategyFactory(strategy):
    if strategy=='dv':
        return Greedy()
    elif strategy =='rs':
        return Sample()
    elif strategy == 'conf':
        return Confidence()
    # elif strategy == 'mix':
    #     return Mix()
    # elif strategy == 'seq':
    #     return Seq()
    else:
        return SeqCLF()