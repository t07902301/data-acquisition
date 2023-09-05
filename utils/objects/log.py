import utils.objects.Config as Config
import utils.objects.Detector as Detector
import os
import torch
from utils.logging import *

class Log():
    root: str
    path: str
    name: str
    def __init__(self, model_config:Config.NewModel, name) -> None:
        '''
        Add log name ('data','indices',...) to the model_config root
        '''
        assert model_config.root_detector != None, 'Set up model config path first'
        self.root = os.path.join(model_config.root_detector, 'log', name)
        Config.check_dir(self.root)
        self.name = name
   
    def set_path(self, acquistion_config:Config.Acquisition):
        self.path = os.path.join(self.root, '{}_{}.pt'.format(acquistion_config.method, acquistion_config.n_ndata))  

    def export(self, acquistion_config:Config.Acquisition, data=None, detector:Detector.Prototype=None):
        self.set_path(acquistion_config)
        if self.name == 'detector':
            detector.save(self.path)        
        else:
            torch.save(data, self.path)
            logger.info('{} log save to {}'.format(self.name, self.path))       

    def import_log(self, operation:Config.Operation, general_config):
        self.set_path(operation.acquisition)
        if self.name == 'detector':
            detector = Detector.factory(operation.detection.name, general_config, operation.detection.vit)
            detector.load(self.path) 
            return detector      
        else:
            logger.info('{} log load from {}'.format(self.name, self.path))       
            return torch.load(self.path)
        