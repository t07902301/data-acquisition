# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# Coarse labels is added for cifar100 as an option 

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

class CIFAR100(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True,
                 transform=None,
                 download=False, coarse_label_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.coarse_label_transform = coarse_label_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = []
            for fentry in self.train_list: # dataset is divided to batches. From each batch, retreive images and labels
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_labels += entry['fine_labels']
                self.train_coarse_labels += entry['coarse_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            self.test_labels = entry['fine_labels']
            self.test_coarse_labels = entry['coarse_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            coarse_target = self.train_coarse_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            coarse_target = self.test_coarse_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.coarse_label_transform is not None:
            if coarse_target in self.coarse_label_transform:
                coarse_target = self.coarse_label_transform[coarse_target]

        return img, coarse_target, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

class Novelty(data.Dataset):
    def __init__(self, dataset, cover_labels):
        super().__init__()
        self.dataset = dataset
        self.labels = cover_labels
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img, coarse_label, fine_label, real_idx = self.dataset[index]
        if fine_label in self.labels:
            novelty = 0
        else:
            novelty = 1
        return img, novelty, index
## TODO base transform makes images tensor and not be further transformed
# class Subset(data.Dataset):
#     r"""
#     Subset of a dataset at specified indices.

#     Args:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#     # dataset: CIFAR10
#     # indices: [int]

#     def __init__(self, dataset, indices, augment=None, coarse_label_transform=None) -> None:
#         self.dataset = dataset
#         self.indices = indices
#         self.augment = augment
#         self.coarse_label_transform = coarse_label_transform
 
#     def __getitem__(self, idx):
#         if isinstance(idx, list):
#             return self.dataset[[self.indices[i] for i in idx]]
#         img, coarse_label, fine_label = self.dataset[self.indices[idx]]
#         if (self.dataset.augment is not None) and (self.augment is not None) :
#             img = Image.fromarray(img)
#             img = self.transform(img)         
#         if self.coarse_label_transform is not None:
#             coarse_label = self.coarse_label_transform[coarse_label]
#         return img, coarse_label, fine_label

#     def __len__(self):
#         return len(self.indices)

    # def set_transform(self, new_transform):
    #     self.transform = new_transform

