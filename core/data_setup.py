from utils.set_up import *

def main(epochs,  model_dir, dataset_name):
    config = load_config(model_dir)
    save_dataset(epochs, config, model_dir, dataset_name)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-ds','--dataset',type=str, default='core')

    args = parser.parse_args()
    main(args.epochs,args.model_dir, args.dataset)