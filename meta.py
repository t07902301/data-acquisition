from utils.dataset.wrappers import MetaData
from utils.set_up import load_config
from utils.logging import *
import numpy as np
import os
import pickle as pkl
def run(model_dir, frame_sample_ratio):
    config = load_config(model_dir)
    meta_data = MetaData('core_data.pkl')
    label_map = config['data']['labels']['map']
    category = list(label_map.keys()) if label_map != None else [i for i in range(10)]
    sessions = [i for i in range(11)]
    all_data_indices = np.arange(len(meta_data.data))
    split_results = meta_data.balanced_split(frame_sample_ratio, category, sessions, all_data_indices) # sample ratio of frames can be float too  
    return meta_data.subset2dict(split_results['sampled'], all_data_indices)

def main(model_dir =''):
    fh = logging.FileHandler('log/{}/meta.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    config = load_config(model_dir)
    frame_sample_ratio = config['data']['ratio']['frame']

    meta = run(model_dir, frame_sample_ratio)
    meta_path = os.path.join('data/meta', '{}.pkl'.format(model_dir[:2]))
    fw = open(meta_path, 'wb')
    pkl.dump(meta, fw)
    print('save to', meta_path)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-md','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.model_dir)