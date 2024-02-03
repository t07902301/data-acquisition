import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))
from utils.set_up import *
import utils.dataset.wrappers as dataset_utils
import matplotlib.pyplot as plt
def main():
    model_dir = 'cifar_4class'
    n_sample = 3
    _, ds_list, normalize_stat = set_up(1, model_dir, 0)
    test_ds = ds_list[0]['test_shift']
    val_ds = ds_list[0]['val_shift']
    target = [3, 43, 97, 66, 75, 34]
    target_names = ['Large Mammals: Bear + Lion + Wolf', 'Medium-sized Mammals: Raccoon + Skunk + Fox']
    src = [3, 43, 66, 75]
    src_names = ['Large Mammals: Bear + Lion', 'Medium-sized Mammals: Raccoon + skunk']
    transform = dataset_utils.get_vis_transform(normalize_stat['mean'], normalize_stat['std'])

    # fig, ax = plt.subplots(len(target), n_sample, figsize=(3,6))
    # ax = ax.flatten()
    # title_idx = 0
    # for row, label in enumerate(target):
    #     subset, _ = dataset_utils.Cifar().get_subset_by_labels(test_ds, [label])
    #     if row % 3 == 0:
    #         ax[row * n_sample + 1].set_title(target_names[title_idx], fontsize=10)
    #         title_idx += 1

    #     for col in range(n_sample):
    #         idx = col + row * n_sample
    #         ax[idx].imshow(transform(subset[col][0]))
    #         ax[idx].axis(False)
    # fig.tight_layout()
    # fig.savefig('figure/target.png')
    # print('save')
    # plt.cla()

    fig, ax = plt.subplots(len(src), n_sample, figsize=(4,5))
    ax = ax.flatten()
    title_idx = 0
    for row, label in enumerate(src):
        subset, _ = dataset_utils.Cifar().get_subset_by_labels(val_ds, [label])
        if row % 2 == 0:
            ax[row * n_sample + 1].set_title(src_names[title_idx], fontsize=13)
            title_idx += 1
        for col in range(n_sample):
            idx = col + row * n_sample
            ax[idx].imshow(transform(subset[col][0]))
            ax[idx].axis(False)
    fig.tight_layout()
    fig.savefig('figure/src.png')
    print('save')
    plt.cla()

    # new = [97,34]
    # new_names = ['Wolf', 'Fox']

    # fig, ax = plt.subplots(len(new), n_sample, figsize=(3,3))
    # ax = ax.flatten()
    # for row, label in enumerate(new):
    #     subset, _ = dataset_utils.Cifar().get_subset_by_labels(test_ds, [label])
    #     for col in range(n_sample):
    #         idx = col + row * n_sample
    #         ax[idx].imshow(transform(subset[col][0]))
    #         ax[idx].axis(False)
    #     title_idx = 0 + row * n_sample
    #     ax[title_idx].set_title(new_names[row])
    # fig.tight_layout()
    # fig.savefig('figure/new.png')
    # print('save')
    # plt.cla()

    # old = [3, 66]
    # old_names = ['Bear', 'Raccoon']

    # # cifar_utils = dataset_utils.Cifar()
    # # n_sample = 3
    # # transform = dataset_utils.get_vis_transform(normalize_stat['mean'], normalize_stat['std'])
    
    # fig, ax = plt.subplots(len(old), n_sample, figsize=(3,3))
    # ax = ax.flatten()
    # for row, label in enumerate(old):
    #     subset, _ = dataset_utils.Cifar().get_subset_by_labels(test_ds, [label])
    #     for col in range(n_sample):
    #         idx = col + row * n_sample
    #         ax[idx].imshow(transform(subset[col][0]))
    #         ax[idx].axis(False)
    #     title_idx = 0 + row * n_sample
    #     ax[title_idx].set_title(old_names[row])
    # fig.tight_layout()
    # fig.savefig('figure/old.png')
    # print('save')
    # plt.cla()

main()
        
    