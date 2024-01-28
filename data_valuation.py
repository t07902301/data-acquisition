from utils.strategy import *
from utils.set_up import *
import utils.statistics.checker as Checker
from typing import List
from utils.logging import *
import matplotlib.pyplot as plt
import utils.statistics.distribution as distribution_utils
import utils.statistics.partitioner as Partitioner

class TestData():
    def __init__(self) -> None:
        pass
   
    def probab2weight(self, dstr_dict: Partitioner.Dict[str, distribution_utils.CorrectnessDisrtibution], observations):
        weights = []
        probab_partitioner = Partitioner.Posterior()
        for value in observations:
            posterior = probab_partitioner.get_posterior(value, dstr_dict)
            weights.append(posterior)
        return np.concatenate(weights)
    
    def plot(self, data_valuation, weakness_score, probab, n_data, color=None, file_name=None, fig_name=None):
        new_data_indices = acquistion.get_top_values_indices(data_valuation, n_data)
        selected_weakness_score = weakness_score[new_data_indices]
        selected_probab_weakness_score = probab[new_data_indices]
        with open(file_name, 'wb') as f:
            pkl.dump({'score': selected_weakness_score, 'probab': selected_probab_weakness_score}, f)
            f.close()
        logger.info('{} saved'.format(file_name))
        # n_data = [ i for i in range(n_data)]    
        # plt.subplot(211)
        # plt.plot(n_data, selected_weakness_score, label='weakness score', color=color)
        # plt.legend(fontsize=12)
        # plt.subplot(212)
        # plt.plot(n_data, selected_probab_weakness_score, label='probability', color=color)
        # plt.legend(fontsize=12)
        # plt.xlabel('the Number of Selected Data', fontsize=12)
        # plt.savefig(fig_name)
        # logger.info(fig_name)
        # plt.clf()

    def run(self, epoch, parse_args:ParseArgs, operation:Config.Operation, dataset_list: List[dict]):
        '''
        Visualize data valuation results of a dataset from a given epoch 
        '''
        logger.info('in epoch {}'.format(epoch))
        checker = Checker.instantiate(epoch, parse_args, dataset_list[epoch], operation)
        anchor_dstr = checker.set_up_dstr(checker.anchor_loader, operation.ensemble.pdf)
        datasplits = dataset_utils.DataSplits(dataset_list[epoch], checker.new_model_config.new_batch_size)
        weakness_score, _ = checker.detector.predict(datasplits.loader['market'])
        new_weight = self.probab2weight({'target': anchor_dstr['incorrect'], 'other': anchor_dstr['correct']}, weakness_score)
        
        data_valuation, file_name = weakness_score, 'log/{}/top_score.pkl'.format(parse_args.model_dir)
        self.plot(data_valuation, weakness_score, new_weight, 100, file_name=file_name)

        # data_valuation, fig_name = new_weight, 'log/{}/top_probab.png'.format(model_dir)
        # self.plot(data_valuation, weakness_score, new_weight, 100, color='blue', fig_name=fig_name)

def main(epochs, model_dir ='', device=0, probab_bound = 0.5, detector_name = ''):
    fh = logging.FileHandler('log/{}/data_valuation.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Detector: {}; Probab bound: {}'.format(detector_name, probab_bound))
    
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    parse_args, ds_list, normalize_stat = set_up(epochs, model_dir, device)

    clip_processor = Detector.load_clip(device_config, normalize_stat['mean'], normalize_stat['std'])
    ensemble_instruction = Config.Ensemble(name='ae-w')
    detect_instruction = Config.Detection(detector_name, clip_processor)
    acquire_instruction = Config.AcquisitionFactory(acquisition_method='', data_config=parse_args.general_config['data'], utility_estimator='u-ws')
    operation = Config.Operation(acquire_instruction, ensemble_instruction, detect_instruction)
    
    stat_checker = TestData()
    stat_checker.run(0, parse_args, operation, ds_list)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help="(dataset name) _ task _ (other info)")
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm', help="svm, logistic regression")

    args = parser.parse_args()
    main(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name)
