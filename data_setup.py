from utils.set_up import *

from utils.logging import *

def main(epochs,  model_dir, save_mode):
    fh = logging.FileHandler('log/{}/data.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    config = load_config(model_dir)

    if save_mode == 'shift':
        save_dataset_shift(epochs, model_dir, config)

    else: 
        save_dataset_split(epochs, model_dir, config)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='', help='model directory')
    parser.add_argument('-sm','--save_mode',type=str, default='shift', help='shist, split: save indices of data shifts or splits')

    args = parser.parse_args()
    main(args.epochs,args.model_dir, args.save_mode)