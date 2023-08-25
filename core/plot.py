from utils.statistics.plot import Line

def dev(epochs, dev_name, device, detector_name, model_dir, stream_name, base_type, option):
    print(stream_name)
    new_model_setter = 'retrain'
    pure = True
    
    Line().run(result, method_list, config['data']['n_new_data'], 'acc change')

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-dn','--detector_name',type=str,default='svm')
    parser.add_argument('-dev','--dev',type=str, default='dv')
    parser.add_argument('-s','--stream',type=str, default='probab')
    parser.add_argument('-bt','--base_type',type=str,default='cnn')
    parser.add_argument('-op','--option',type=str, default='object')

    args = parser.parse_args()
    dev(args.epochs, model_dir=args.model_dir, device=args.device, detector_name=args.detector_name, dev_name=args.dev, stream_name=args.stream, base_type=args.base_type, option= args.option)
