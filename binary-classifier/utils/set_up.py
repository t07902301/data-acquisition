import utils.objects.Config as Config
import utils.objects.dataset as Dataset
import torch
def set_up(epochs,  pure, device=0):
    batch_size, train_labels, target_test_labels, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config = Config.parse(pure)
    Config.display(batch_size, label_map, new_img_num_list, superclass_num, ratio)
    ds_list = Dataset.get_data_splits_list(epochs, train_labels, label_map, ratio, target_test_labels)
    # ds_list = Dataset.get_data_splits_list(epochs, train_labels, label_map, ratio)
    device_config = 'cuda:{}'.format(device)
    torch.cuda.set_device(device_config)
    return batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config
