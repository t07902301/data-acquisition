import utils.objects.Config as Config
import torch
def get_config(model_config:Config.NewModel):
    log_config = Config.Log(batch_size=model_config.batch_size,class_number=model_config.class_number,model_dir=model_config.model_dir,pure=model_config.pure,setter=model_config.setter,model_cnt=model_config.model_cnt, device=model_config.device, augment=model_config.augment)
    return log_config

def get_sub_log(log_symbol, model_config:Config.NewModel, acquisition_config:Config.Acquistion):
    log_config = get_config(model_config)
    log_config.set_sub_log_root(log_symbol)
    log_config.set_path(acquisition_config)    
    return log_config

# def log_max_dv(model_cofig:Config.NewModel, acquisition_config:Config.Acquistion, dv_list):
#     log_config = Config.Log(batch_size=model_cofig.batch_size,class_number=model_cofig.class_number,model_dir=model_cofig.model_dir, model_cnt=model_cofig.model_cnt,pure=model_cofig.pure,setter=model_cofig.setter)
#     log_config.root = os.path.join(log_config.root, 'dv')
#     log_config.check_dir(log_config.root)
#     log_config.set_path(acquisition_config)
#     torch.save(np.max(dv_list), log_config.path)
#     print('save Max dv to', log_config.path)

def load(log_config:Config.Log):
    data = torch.load(log_config.path)
    print('{} log load from {}'.format(log_config.sub_log_symbol, log_config.path))
    return data

def save(data, log_config:Config.Log):
    if log_config.sub_log_symbol == 'data' and log_config.augment:
        print('Cannot save augmented data')
    else:
        torch.save(data, log_config.path)
        print('{} log save to {}'.format(log_config.sub_log_symbol, log_config.path))    
