import sys
sys.path.append('/home/yiwei/data-acquisition/binary/')
# sys.path.append('..')
from utils.strategy import *
import utils.statistics.checker as Checker

def method_run(n_img_list, acquisition_config:Config.Acquistion, checker: Checker.prototype):
    acc_change_list = []
    for n_img in n_img_list:
        acquisition_config.n_ndata = n_img
        acc_change = n_data_run(acquisition_config, checker)
        acc_change_list.append(acc_change)
    return acc_change_list

def n_data_run(acquisition_config:Config.Acquistion, checker: Checker.prototype):
    check_result = checker.run(acquisition_config)
    return check_result

def run(acquisition_config:Config.Acquistion, methods, new_img_num_list, checker:Checker.prototype):
    result_list = []
    for method in methods:
        print('In method', method)
        acquisition_config.method = method
        result_method = method_run(new_img_num_list, acquisition_config, checker)
        result_list.append(result_method)
    return result_list

# def check_bound(old_model_config: Config.OldModel, datasplit: Dataset.DataSplits, acquire_instruction: Config.Acquistion, clip_processor):
#     base_model = Model.resnet(2)
#     base_model.load(old_model_config)
#     clf = Detector.SVM(datasplit.loader['train_clip'], clip_processor)
#     _ = clf.fit(base_model, datasplit.loader['val_shift'])
#     market_dv, _ = clf.predict(datasplit.loader['market'])
#     return (market_dv <= acquire_instruction.bound).sum()

def epoch_run(new_img_num_list, method_list, acquire_instruction:Config.Acquistion, checker: Checker.prototype):
    result_epoch = run(acquire_instruction, method_list, new_img_num_list, checker)
    # bound_stat = check_bound(old_model_config, dataset_splits,acquire_instruction, clip_processor)
    return result_epoch, 0

def get_checker_args(epoch, parse_args, dataset):
    batch_size, superclass_num,model_dir, device_config, base_type, pure, new_model_setter, seq_rounds_config = parse_args
    old_model_config = Config.OldModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, base_type=base_type)
    new_model_config = Config.NewModel(batch_size['base'], superclass_num, model_dir, device_config, epoch, pure, new_model_setter, batch_size['new'], base_type=base_type)
    dataset_splits = Dataset.DataSplits(dataset, old_model_config.batch_size)
    return old_model_config, new_model_config, dataset_splits

def get_checker(epoch, parse_args, dataset, stream_instruction:Config.Stream, detect_instruction: Config.Dectector, plot=True):
    old_model_config, new_model_config, dataset_splits = get_checker_args(epoch, parse_args, dataset)
    checker = Checker.factory(stream_instruction.name, new_model_config)
    checker.setup(old_model_config, dataset_splits, detect_instruction, stream_instruction, plot)
    return checker