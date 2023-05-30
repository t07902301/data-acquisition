import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import utils.objects.Detector as Detector
import torch

# def get_config(model_config:Config.NewModel, acquisition_config:Config.Acquistion, log_symbol):
#     log_config = Config.Log(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter,model_cnt=model_config.model_cnt, device=model_config.device, augment=model_config.augment, new_batch_size=model_config.new_batch_size, log_symbol=log_symbol, type=model_config.model_type)
#     log_config.set_path(acquisition_config)    
#     return log_config

# def log_max_dv(model_cofig:Config.NewModel, acquisition_config:Config.Acquistion, dv_list):
#     log_config = Config.Log(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
#     log_config.root = os.path.join(log_config.root, 'dv')
#     log_config.check_dir(log_config.root)
#     log_config.set_path(acquisition_config)
#     torch.save(np.max(dv_list), log_config.path)
#     print('save Max dv to', log_config.path)

def load(log_config:Config.Log):
    data = torch.load(log_config.path)
    print('{} log load from {}'.format(log_config.log_symbol, log_config.path))
    return data

def save(data, log_config:Config.Log, augment=False):
    if log_config.log_symbol == 'data' and augment:
        print('Cannot save augmented data')
    else:
        torch.save(data, log_config.path)
        print('{} log save to {}'.format(log_config.log_symbol, log_config.path))    

def get_log_data(acquisition_config:Config.Acquistion, model_config:Config.NewModel, dataset_splits:Dataset.DataSplits):
    if 'seq' in acquisition_config.method:
        log_symbol = 'data'
    else:
        log_symbol = 'indices'

    log_config = model_config.get_log_config(log_symbol)
    log_config.set_path(acquisition_config)
    log_content = load(log_config)  

    if log_symbol == 'indices':
        new_data_indices = log_content
        new_data = torch.utils.data.Subset(dataset_splits.dataset['market'], new_data_indices)
    else:
        new_data = log_content

    return new_data

def get_log_clf(acquisition_config:Config.Acquistion, model_config:Config.NewModel, set_up_dataloader, clip_processor):
    log_config = model_config.get_log_config('clf')
    log_config.set_path(acquisition_config)
    detector = Detector.SVM(set_up_dataloader, clip_processor)
    clf = torch.load(log_config.path, map_location=model_config.device)
    detector.fitter.clf = clf
    print('{} log load from {}'.format(log_config.log_symbol, log_config.path))
    return detector