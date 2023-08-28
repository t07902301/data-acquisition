from utils.dataset.wrappers import MetaData
from utils.set_up import load_config
import os
import pickle as pkl
def run(model_dir):
    config = load_config(model_dir)
    meta_data = MetaData('core_data.pkl')
    label_map = config['data']['label_map']
    category = list(label_map.keys()) if label_map != None else [i for i in range(10)]
    sessions = [i for i in range(11)]
    split_results = meta_data.balanced_split(30, category, sessions)
    return meta_data.subset2dict(split_results['sampled'])

def main(model_dir =''):
    meta = run(model_dir)
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