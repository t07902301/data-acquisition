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
from utils.env import data_env
import utils.ood as OOD
import utils.objects.dataloader as dataloader_utils
from utils.logging import logger
from typing import Dict

#TODO add info class with base model and dataset, make strategy class more purified. 

class WorkSpace():
    '''
    Data + Base Model: reset before each strategy operation
    '''
    def __init__(self, model_config: Config.OldModel, dataset:dict, config) -> None:
        self.base_model_config = model_config
        self.init_dataset = dataset
        self.general_config = config
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
   
    def set_validation(self, new_batch_size):
        gt, pred, _ = self.base_model.eval(self.data_split.loader['val_shift'])
        incorrect_mask = (gt != pred)
        incorrect_indices = np.arange(len(self.data_split.dataset['val_shift']))[incorrect_mask]
        incorrect_val = torch.utils.data.Subset(self.data_split.dataset['val_shift'], incorrect_indices)
        self.validation_loader = torch.utils.data.DataLoader(incorrect_val, new_batch_size)

    def set_anchor_dstr(self, pdf):
        correct_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], pdf, correctness=True)
        incorrect_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], pdf, correctness=False)
        self.anchor_dstr = {'correct': correct_dstr, 'incorrect': incorrect_dstr}
        logger.info('Set up anchor dstr')

    # def set_validation(self, stream_instruction:Config.ProbabEnsemble, old_batch_size, new_batch_size, set_anchor_dstr=False):
    #     '''
    #     align validation_loader with test splits\n
    #     after Error detector construction or updates (seq)
    #     '''
    #     correct_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=True)
    #     incorrect_dstr = distribution_utils.CorrectnessDisrtibution(self.base_model, self.detector, self.data_split.loader['val_shift'], stream_instruction.pdf, correctness=False)

    #     val_shift_info = DataStat.build_info(self.data_split, 'val_shift', self.detector, old_batch_size, new_batch_size)
    #     val_shift_split, _ = Partitioner.Probability().run(val_shift_info, {'target': incorrect_dstr, 'other': correct_dstr}, stream_instruction)
    #     self.validation_loader = val_shift_split['new_model']
    #     logger.info('set validation_loader') # Align with test set inference

    #     if set_anchor_dstr:
    #         self.anchor_dstr = {'correct': correct_dstr, 'incorrect': incorrect_dstr}
    #         logger.info('Set up anchor dstr')

    def reset(self, new_batch_size, clip_processor):
        '''
        Reset Model and Data if Model is modifed after each operation
        '''
        self.set_up(new_batch_size, clip_processor)

    def set_market(self, clip_processor, known_labels):
        filtered_market = OOD.run(self.data_split, clip_processor, known_labels, check_ds='market')
        self.init_dataset['market'] = filtered_market
        # cover_aug_market_dataset = OOD.run(self.data_split, clip_processor, known_labels, check_ds='aug_market')
        # self.init_dataset['aug_market'] = cover_aug_market_dataset
        logger.info('After filtering, Market size:{}'.format(len(self.init_dataset['market'])))

    def set_detector(self, detector_instruction: Config.Detection):
        detector = Detector.factory(detector_instruction.name, self.general_config, detector_instruction.vit)
        detector.fit(self.base_model, self.data_split.loader['val_shift'])
        self.detector = detector

    def reset_detector(self, detector_instruction: Config.Detection):
        self.set_detector(detector_instruction)
    
class Strategy():
    def __init__(self) -> None:
        self.stat_mode = False
        data_env() # random seed set up

    @abstractmethod
    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        '''
        Under an acquistion instruction (which method, how much new data), this strategy operates on data and the old Model. Finally, the new model will be saved. \n
        Init workspace should be identical across every method and budget
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

        # if workspace.base_model_config.base_type == 'cnn':

        #     new_data = torch.utils.data.Subset(workspace.data_split.dataset['aug_market'], new_data_indices)
        
        # else:
        #     new_data = torch.utils.data.Subset(workspace.data_split.dataset['market'], new_data_indices)
        
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

    def export_indices(self, model_config:Config.NewModel, acquire_instruction: Config.Acquisition, dataset, ensemble: Config.Ensemble):
        if acquire_instruction.method != 'rs':
            raw_indices = [dataset[idx][-1] for idx in range(len(dataset))]
            log = Log(model_config, 'indices')
            log.export(acquire_instruction, data=raw_indices)

class NonSeqStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, budget, market_loader):
        pass

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)

        new_data_info = self.get_new_data_info(operation, workspace)

        self.update_dataset(new_model_config, workspace, operation.acquisition, new_data_info['data'])

        new_model_config.set_path(operation)
        logger.info(new_model_config.path)

        self.update_model(new_model_config, workspace)

        workspace.base_model.save(new_model_config.path)

        self.export_indices(new_model_config, operation.acquisition, new_data_info['data'], operation.stream)
        
class Greedy(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()

    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_budget = operation.acquisition.budget
        new_data_indices = self.run(acquistion_budget, dataset_splits.loader['market'], workspacce.detector, operation.acquisition.threshold)
        return new_data_indices  
    
    def run(self, budget, dataloader, detector: Detector.Prototype, threshold):
        utility_score, _ = detector.predict(dataloader)
        if threshold == None:
            top_data_indices = acquistion.get_top_values_indices(utility_score, budget)
        else:
            # top_data_indices = acquistion.get_top_indices_threshold(utility_score, budget, threshold)
            top_data_indices = acquistion.get_threshold_indices(utility_score, threshold)
        return top_data_indices

class ProbabGreedy(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()

    def probability_utility(self, dstr_dict: Dict[str, distribution_utils.CorrectnessDisrtibution], observations):
        probabilities = []
        probab_partitioner = Partitioner.Probability()
        for value in observations:
            posterior = probab_partitioner.get_posterior(value, dstr_dict)
            probabilities.append(posterior)
        return np.concatenate(probabilities)
    
    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_budget = operation.acquisition.budget
        new_data_indices = self.run(acquistion_budget, dataset_splits.loader['market'], workspacce.detector, workspacce.anchor_dstr, operation.acquisition.threshold)
        return new_data_indices  
    
    def run(self, budget, dataloader, detector: Detector.Prototype, anchor_dstr, threshold=None):
        feature_score, _ = detector.predict(dataloader)
        utility_score = self.probability_utility({'target': anchor_dstr['incorrect'], 'other': anchor_dstr['correct']}, feature_score)
        if threshold == None:
            top_data_indices = acquistion.get_top_values_indices(utility_score, budget)
        else:
            # top_data_indices = acquistion.get_top_indices_threshold(utility_score, budget, threshold)
            top_data_indices = acquistion.get_threshold_indices(utility_score, threshold)

        # select_utility_score = market_utility_score[new_data_indices]
        # select_probab_utility_score = utility_score[new_data_indices]
        # logger.info('dv: {}, {}'.format(min(select_utility_score), max(select_utility_score)))
        # logger.info('dv probab: {}, {}'.format(min(select_probab_utility_score), max(select_probab_utility_score)))
        return top_data_indices

class Sample(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()

    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_budget = operation.acquisition.budget
        new_data_indices = self.run(acquistion_budget, dataset_splits.dataset['market'])
        return new_data_indices      
    
    def run(self, budget, dataset):
        # Market sampling does not know data subclass labels which are often inaccessible
        dataset_indices = np.arange(len(dataset))
        sample_indices = acquistion.sample(dataset_indices, budget)
        return sample_indices

class Confidence(NonSeqStrategy):
    def __init__(self) -> None:
        super().__init__()
    
    def get_new_data_indices(self, operation:Config.Operation, workspacce:WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_budget = operation.acquisition.budget
        new_data_indices = self.run(acquistion_budget, dataset_splits.loader['market'], workspacce.base_model, workspacce.base_model_config)
        return new_data_indices      

    def get_confidence_score(self,  base_model_config: Config.OldModel, base_model:Model.Prototype,market_loader):

        market_gts, _, market_score = base_model.eval(market_loader)

        if base_model_config.base_type == 'svm':
            confs = acquistion.get_gt_distance(market_gts, market_score)
        else:
            confs = acquistion.get_gt_probab(market_gts, market_score)
        return confs

    def run(self, budget, dataloader,  base_model:Model.Prototype, base_model_config: Config.OldModel):

        confs = self.get_confidence_score(base_model_config, base_model, dataloader)

        top_data_indices = acquistion.get_top_values_indices(confs, budget, order='ascend')

        return top_data_indices

class Mix(Sample):
    def __init__(self) -> None:
        super().__init__()

    def run(self, budget, base_model: Model.Prototype, data_loader):
        gt, pred, _ = base_model.eval(data_loader)
        incorrect_mask = (gt!=pred)
        incorect_indices = np.arange(len(gt))[incorrect_mask]
        return acquistion.sample(incorect_indices, budget)

    def get_new_data_indices(self, operation: Config.Operation, workspacce: WorkSpace):
        dataset_splits = workspacce.data_split
        acquistion_budget = operation.acquisition.budget
        new_data_indices = self.run(acquistion_budget, workspacce.base_model, dataset_splits.loader['market'])
        return new_data_indices 

    # def get_new_data_indices(self, budget, dataset_splits: dataset_utils.DataSplits, detector_instruction: Config.Detection, threshold = None):
    #     detector = Detector.SVM(dataset_splits.loader['train_clip'], self.clip_processor)
    #     _ = detector.fit(self.base_model, dataset_splits.loader['val_shift'])
    #     market_utility_score, _ = detector.predict(dataset_splits.loader['market'], self.base_model)
    #     greedy_results = acquistion.get_top_values_indices(market_utility_score, budget-budget//2)
    #     sample_results = acquistion.sample_acquire(market_utility_score,budget//2)
    #     new_data_cls_indices = np.concatenate([greedy_results, sample_results])
    #     detector_info = None
    #     return new_data_cls_indices, detector_info       

class SeqCLF(Strategy):
    '''
    Detector and Validation Shift are updated after each operation. 
    '''
    def __init__(self) -> None:
        super().__init__()

    def operate(self, operation: Config.Operation, new_model_config:Config.NewModel, workspace:WorkSpace):
        workspace.reset(new_model_config.new_batch_size, operation.detection.vit)
        workspace.reset_detector(operation.detection)
        operation.acquisition.set_up()

        self.sub_strategy = StrategyFactory(operation.acquisition.round_acquire_method)
        new_data_total_set = None

        stat_results = []

        for round_i in range(operation.acquisition.n_rounds):
            new_data_round_info = self.round_operate(round_i, operation, workspace)
            new_data_total_set = new_data_round_info['data'] if round_i==0 else torch.utils.data.ConcatDataset([new_data_total_set, new_data_round_info['data']])
            if self.stat_mode:
                _, acc = workspace.detector.predict(workspace.data_split.loader['test_shift'], workspace.base_model, metrics='precision')
                stat_results.append(acc) # detector predcision change

                # stat_results.append(self.error_stat(workspace.base_model, new_data_round_info['data'], new_model_config.new_batch_size)) # error in acquired data 

                # stat_results.append(self.error_stat(workspace.base_model, workspace.data_split.dataset['val_shift'], new_model_config.new_batch_size)) # error to train detector
        
        if self.stat_mode:
            return stat_results
        
        new_model_config.set_path(operation)

        workspace.set_data(new_model_config.new_batch_size) # recover validation & (aug)market
        workspace.set_validation(new_model_config.new_batch_size)

        self.update_dataset(new_model_config, workspace, operation.acquisition, new_data_total_set)

        self.update_model(new_model_config, workspace)

        workspace.base_model.save(new_model_config.path)

        self.export_detector(new_model_config, operation.acquisition, workspace.detector)
        self.export_indices(new_model_config, operation.acquisition, new_data_total_set, operation.stream)

    def error_stat(self, model: Model.Prototype, data, batch_size):
        dataloader = torch.utils.data.DataLoader(data, batch_size)
        acc = model.acc(dataloader)
        return 100 - acc

    def round_operate(self, round_id, operation: Config.Operation, workspace:WorkSpace):
        '''
        Get new data, update (aug)market and val_shift, new detector from updated val_shift
        '''
        round_operation = self.round_set_up(operation, round_id)    

        new_data_round_info = self.sub_strategy.get_new_data_info(round_operation, workspace)
        logger.info('In round {}, {} data is acquired'.format(round_id, len(new_data_round_info['indices'])))

        workspace.data_split.reduce('market', new_data_round_info['indices'])
        workspace.data_split.expand('val_shift', new_data_round_info['data'])

        workspace.set_detector(operation.detection)

        return new_data_round_info

    def round_set_up(self, operation: Config.Operation, round_id):
        '''
        Set up data in each round (by round ID)
        '''
        round_operation = deepcopy(operation)
        if operation.acquisition.seq_mode == 'budget':
            round_operation.acquisition.budget = operation.acquisition.budget_round
            return round_operation
        else:
            round_operation.acquisition.current_round = round_id + 1
            if round_operation.acquisition.current_round == round_operation.acquisition.n_rounds:
                round_operation.acquisition.budget = operation.acquisition.budget_last_round
            else:
                round_operation.acquisition.budget = operation.acquisition.budget_non_last_round
            return round_operation

    def export_detector(self, model_config:Config.NewModel, acquire_instruction: Config.SequentialAc, detector: Detector.Prototype):
        log = Log(model_config, 'detector')
        log.export(acquire_instruction, detector=detector)
        
    def get_new_data_indices(self, operation:Config.Operation, workspace:WorkSpace):
        pass

class SeqPD(SeqCLF):
    def __init__(self) -> None:
        super().__init__()

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

#         assert len(new_data_total_set) == acquire_instruction.budget, 'size error with new data'
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
    elif strategy == 'pd':
        return ProbabGreedy()
    elif strategy == 'seq_pd':
        return SeqPD()
    elif strategy == 'mix':
        return Mix()
    # elif strategy == 'seq':
    #     return Seq()
    else:
        return SeqCLF()