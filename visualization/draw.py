from utils.set_up import *
from utils.logging import *
import utils.dataset.wrappers as dataset_utils
import matplotlib.pyplot as plt
def main():
    model_dir = 'c5'
    fh = logging.FileHandler('log/draw.log'.format(model_dir),mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    config, device_config, ds_list, normalize_stat = set_up(1, model_dir, 0, '', 'cifar')
    test_ds = ds_list[0]['test_shift']
    train_ds = ds_list[0]['train']

    target = [13, 85, 90, 81]
    target_names = ['class 0: bus', 'class 1: tank', 'class 0: train', 'class 1: streetcar']
    src = [13, 85]
    src_names = ['class 0: bus', 'class 1 : tank']

    new = [90, 81]
    new_names = ['train', 'streetcar']

    cifar_utils = dataset_utils.Cifar()
    n_sample = 3
    transform = dataset_utils.get_vis_transform(normalize_stat['mean'], normalize_stat['std'])
    
    fig, ax = plt.subplots(len(new), n_sample, figsize=(3,3))
    ax = ax.flatten()
    for row, label in enumerate(new):
        subset, _ = cifar_utils.get_subset_by_labels(test_ds, [label])
        for col in range(n_sample):
            idx = col + row * n_sample
            ax[idx].imshow(transform(subset[col][0]))
            ax[idx].axis(False)
        title_idx = 0 + row * n_sample
        ax[title_idx].set_title(new_names[row])
    plt.suptitle('Class C_w')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figure/new.png')
    logger.info('save')
    plt.cla()

    old = [13, 85]
    old_names = ['bus', 'tank']

    # cifar_utils = dataset_utils.Cifar()
    # n_sample = 3
    # transform = dataset_utils.get_vis_transform(normalize_stat['mean'], normalize_stat['std'])
    
    fig, ax = plt.subplots(len(old), n_sample, figsize=(3,3))
    ax = ax.flatten()
    for row, label in enumerate(old):
        subset, _ = cifar_utils.get_subset_by_labels(test_ds, [label])
        for col in range(n_sample):
            idx = col + row * n_sample
            ax[idx].imshow(transform(subset[col][0]))
            ax[idx].axis(False)
        title_idx = 0 + row * n_sample
        ax[title_idx].set_title(old_names[row])
    plt.suptitle('Class C_w \'')
    plt.tight_layout()
    # plt.show()
    plt.savefig('figure/old.png')
    logger.info('save')
    plt.cla()

    # fig, ax = plt.subplots(len(target), n_sample, figsize=(n_sample,4))
    # ax = ax.flatten()
    # for row, label in enumerate(target):
    #     subset, _ = cifar_utils.get_subset_by_labels(test_ds, [label])
    #     for col in range(n_sample):
    #         idx = col + row * n_sample
    #         ax[idx].imshow(transform(subset[col][0]))
    #         ax[idx].axis(False)
    #     title_idx = 1 + row * n_sample
    #     ax[title_idx].set_title(target_names[row])
    # # plt.suptitle('')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('figure/target.png')
    # logger.info('save')
    # plt.cla()

    # n_sample = 3
    # fig, ax = plt.subplots(len(src), n_sample, figsize=(3, 3))
    # ax = ax.flatten()
    # for row, label in enumerate(src):
    #     subset, _ = cifar_utils.get_subset_by_labels(train_ds, [label])
    #     for col in range(n_sample):
    #         idx = col + row * n_sample
    #         ax[idx].imshow(transform(subset[col][0]))
    #         ax[idx].axis(False)
    #     title_idx = 1 + row * n_sample
    #     ax[title_idx].set_title(src_names[row])
    # # plt.suptitle('')
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('figure/src.png')
    # logger.info('save')

main()
        
    