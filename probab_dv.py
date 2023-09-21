from utils.strategy import *
from utils.set_up import *
import utils.statistics.checker as Checker
from typing import List
from utils.logging import *
import matplotlib.pyplot as plt

class TestData():
    def __init__(self) -> None:
        pass
   
    def probab2weight(self, dstr_dict: Dict[str, distribution_utils.CorrectnessDisrtibution], observations):
        weights = []
        probab_partitioner = Partitioner.Probability()
        for value in observations:
            posterior = probab_partitioner.get_posterior(value, dstr_dict)
            weights.append(posterior)
        return np.concatenate(weights)
        
    def run(self, epochs, parse_args, operation:Config.Operation, dataset_list: List[dict]):
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation)
            anchor_dstr = checker.set_up_dstr(checker.anchor_loader, operation.stream.pdf)
            datasplits = dataset_utils.DataSplits(dataset_list[epo], checker.new_model_config.new_batch_size)

            test_dv, _ = checker.detector.predict(datasplits.loader['test_shift'])
            new_weight = self.probab2weight({'target': anchor_dstr['incorrect'], 'other': anchor_dstr['correct']}, test_dv)
            new_data_indices = acquistion.get_top_values_indices(new_weight, 100, order='descend')
            # new_data_indices = acquistion.get_top_values_indices(test_dv, 100)
            select_dv = test_dv[new_data_indices]
            select_probab_dv = new_weight[new_data_indices]    
            n_data = [ i for i in range(100)]    
            plt.subplot(211)
            plt.plot(n_data, select_dv, label='dv')
            plt.legend()
            plt.subplot(212)
            plt.plot(n_data, select_probab_dv, label='probab_dv')
            plt.legend()
            plt.savefig('figure/pd.png')
            logger.info('figure/pd.png')

def main(epochs, new_model_setter='retrain', model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = '', opion = '', dataset_name = '', stat_data='train', dev_name= 'dv'):
    pure = True
    fh = logging.FileHandler('log/pd.log'.format(model_dir, dev_name, stat_data),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Detector: {}; Probab bound: {}'.format(detector_name, probab_bound))
    
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    config, device_config, ds_list, normalize_stat = set_up(epochs, model_dir, device, opion, dataset_name)
    method = dev_name

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    stream_instruction = Config.ProbabStream(bound=probab_bound, pdf='kde', name='probab')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.Acquisition()
    operation = Config.Operation(acquire_instruction, stream_instruction, detect_instruction)
    
    parse_args = (model_dir, device_config, base_type, pure, new_model_setter, config)
    
    stat_checker = TestData()
    stat_checker.run(epochs, parse_args, operation, ds_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-pb','--probab_bound', type=Config.str2float, default=0.5)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-ds','--dataset',type=str, default='core')
    parser.add_argument('-op','--option',type=str, default='object')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-stat','--stat',type=str, default='train')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name, opion=args.option, dataset_name=args.dataset, dev_name=args.dev, stat_data=args.stat)
