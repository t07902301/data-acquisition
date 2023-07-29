from utils.set_up import *

def main(epochs,  model_dir =''):
    batch_size, train_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse()
    save_dataset(epochs, train_labels, label_map, ratio, model_dir[:2])

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.epochs,args.model_dir)