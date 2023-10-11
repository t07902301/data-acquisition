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
    
    def plot(self, data_valuation, feature_scores, probab, fig_name, n_data, color):
        new_data_indices = acquistion.get_top_values_indices(data_valuation, n_data)
        select_feature_score = feature_scores[new_data_indices]
        select_probab_feature_score = probab[new_data_indices]    
        n_data = [ i for i in range(n_data)]    
        plt.subplot(211)
        plt.plot(n_data, select_feature_score, label='feature score', color=color)
        plt.legend(fontsize=12)
        plt.subplot(212)
        plt.plot(n_data, select_probab_feature_score, label='probability', color=color)
        plt.legend(fontsize=12)
        plt.xlabel('the Number of Selected Data', fontsize=12)
        plt.savefig(fig_name)
        logger.info(fig_name)
        plt.clf()

    def run(self, epochs, parse_args, operation:Config.Operation, dataset_list: List[dict]):
        for epo in range(epochs):
            logger.info('in epoch {}'.format(epo))
            checker = Checker.instantiate(epo, parse_args, dataset_list[epo], operation)
            anchor_dstr = checker.set_up_dstr(checker.anchor_loader, operation.stream.pdf)
            datasplits = dataset_utils.DataSplits(dataset_list[epo], checker.new_model_config.new_batch_size)

            feature_scores, _ = checker.detector.predict(datasplits.loader['test_shift'])
            new_weight = self.probab2weight({'target': anchor_dstr['incorrect'], 'other': anchor_dstr['correct']}, feature_scores)
           
            data_valuation, fig_name = feature_scores, 'figure/u-wfs.png'
            self.plot(data_valuation, feature_scores, new_weight, fig_name, 100, 'purple')

            data_valuation, fig_name = new_weight, 'figure/u-wfsd.png' 
            self.plot(data_valuation, feature_scores, new_weight, fig_name, 100, 'blue')

def main(epochs, new_model_setter='retrain', model_dir ='', device=0, probab_bound = 0.5, base_type='', detector_name = ''):
    pure = True
    fh = logging.FileHandler('log/{}/data_valuation.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Detector: {}; Probab bound: {}'.format(detector_name, probab_bound))
    
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    config, device_config, ds_list, normalize_stat, dataset_name, option = set_up(epochs, model_dir, device)

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
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name) _ task _ (other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")
    parser.add_argument('-bt','--base_type',type=str,default='cnn', help="Source/Base Model Type: cnn, svm; structure of cnn is indicated in the arch_type field in config.yaml")
    parser.add_argument('-pd','--probab_bound',type=float,default=0.5, help='A bound of the probability from Cw to assign test set and create corresponding val set for model training.')

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, probab_bound=args.probab_bound, base_type=args.base_type, detector_name=args.detector_name)
