from utils.set_up import *

def main(epochs,  model_dir =''):
    config = load_config(model_dir)
    save_dataset(epochs, config, model_dir)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.epochs,args.model_dir)