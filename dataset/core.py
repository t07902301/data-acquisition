# This is a modified version of original  https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
# Coarse categorys is added for cifar100 as an option 

from PIL import Image
import os
import os.path
import pickle
import torch.utils.data as data

class Core(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, meta_data:dict,
                 transform=None, label_transform=None):
        self.transform = transform
        self.label_transform = label_transform
        
        # now load the picked numpy arrays
        self.data, self.category, self.object, self.session = meta_data['data'], meta_data['category'], meta_data['object'], meta_data['session']

    def __getitem__(self, index):
        img, category, obj, session= self.data[index], self.category[index], self.object[index], self.session[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            category = self.label_transform[category]
        return img, category, obj, session, index
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        return True

class ModifiedDataset(data.Dataset):
    def __init__(self, dataset, category_transform=None):
        super().__init__()
        self.dataset = dataset
        # self.transform = transform
        self.category_transform = category_transform
    # def __iter__(self):
        # worker_info = data.get_worker_info()
        # data_size = len(self.dataset)
        # iter_dataset = []
        # for idx in range(data_size):
        #     img, coarse_category, fine_category = self.dataset[idx]
        #     if self.category_transform is not None:
        #         coarse_category = self.category_transform[coarse_category]            
        #     iter_dataset.append((img, coarse_category, fine_category))
        # if worker_info is not None:  # in one worker process, split workload
        #     per_worker = int(math.ceil(data_size/ float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = 0 + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        #     return iter(iter_dataset[iter_start:iter_end])
        # else:
        #     return iter(iter_dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img, category, obj, session, real_idx = self.dataset[index]
        category = self.category_transform[category]
        return img, category, obj, session, real_idx

class Novelty(data.Dataset):
    def __init__(self, dataset, cover_categorys):
        super().__init__()
        self.dataset = dataset
        self.categorys = cover_categorys
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img, coarse_category, fine_category, real_idx = self.dataset[index]
        if fine_category in self.categorys:
            novelty = 0
        else:
            novelty = 1
        return img, novelty, index
