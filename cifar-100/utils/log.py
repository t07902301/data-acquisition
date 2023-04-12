from utils.Config import *
import torch
def get_log_config(model_config:NewModelConfig):
    log_config = LogConfig(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter,model_cnt=model_config.model_cnt, device=model_config.device, augment=model_config.augment)
    return log_config

def get_sub_log(log_symbol, model_config:NewModelConfig, acquisition_config:AcquistionConfig):
    log_config = get_log_config(model_config)
    log_config.set_sub_log_root(log_symbol)
    log_config.set_path(acquisition_config)    
    return log_config

# def log_max_dv(model_cofig:NewModelConfig, acquisition_config:AcquistionConfig, dv_list):
#     log_config = LogConfig(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
#     log_config.root = os.path.join(log_config.root, 'dv')
#     log_config.check_dir(log_config.root)
#     log_config.set_path(acquisition_config)
#     torch.save(np.max(dv_list), log_config.path)
#     print('save Max dv to', log_config.path)

def load_log(log_config:LogConfig):
    data = torch.load(log_config.path)
    print('{} log load from {}'.format(log_config.sub_log_symbol, log_config.path))
    return data

def save_log(data, log_config:LogConfig):
    if log_config.sub_log_symbol == 'data' and log_config.augment:
        print('Cannot save augmented data')
    else:
        torch.save(data, log_config.path)
        print('{} log save to {}'.format(log_config.sub_log_symbol, log_config.path))    
