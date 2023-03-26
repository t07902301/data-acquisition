from utils.Config import *
import torch
def log_data(data, model_cofig:NewModelConfig, acquisition_config):
    log_config = LogConfig(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
    log_config.set_path(acquisition_config)
    torch.save(data, log_config.path)

def log_indices(indices, model_cofig:NewModelConfig, acquisition_config:AcquistionConfig):
    if acquisition_config.method=='dv':
        log_config = LogConfig(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
        log_config.root = os.path.join(log_config.root, 'indices')
        log_config.check_dir(log_config.root)
        log_config.set_path(acquisition_config)
        torch.save(indices, log_config.path)
        print('save indices log to', log_config.path)

# def log_max_dv(model_cofig:NewModelConfig, acquisition_config:AcquistionConfig, dv_list):
#     log_config = LogConfig(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
#     log_config.root = os.path.join(log_config.root, 'dv')
#     log_config.check_dir(log_config.root)
#     log_config.set_path(acquisition_config)
#     torch.save(np.max(dv_list), log_config.path)
#     print('save Max dv to', log_config.path)

def load_log(path):
    data = torch.load(path)
    print('load log from {}'.format(path))
    return data
def save_log(data, path):
    torch.save(data, path)
    print('save log to', path)