from utils.set_up import *

def main(epochs,  model_dir =''):
    batch_size, total_labels, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse()
    select_fine_labels = total_labels['select_fine_labels']
    label_map = total_labels['label_map']
    task_id = model_dir[:2]
    save_dataset(epochs, select_fine_labels, label_map, ratio, task_id)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')

    args = parser.parse_args()
    main(args.epochs,args.model_dir)