import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
def set_up(epochs,  pure, device_id=0):
    batch_size, train_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse(pure)
    ds_list = Dataset.get_data_splits_list(epochs, train_labels, label_map, ratio)
    device_config = 'cuda:{}'.format(device_id)
    torch.cuda.set_device(device_config)
    return batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config
