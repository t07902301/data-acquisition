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
                 img_transform=None, label_transform=None):
        self.img_transform = img_transform
        self.label_transform = label_transform
        
        # now load the picked numpy arrays
        self.data, self.category, self.object, self.session = meta_data['data'], meta_data['category'], meta_data['object'], meta_data['session']

    def __getitem__(self, index):
        img, category, obj, session= self.data[index], self.category[index], self.object[index], self.session[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            category = self.label_transform[category]
        return img, category, obj, session, index
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        return True

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

name2cat = {
    "plug_adapter": 0,
    "mobile_phone": 1,
    "scissor": 2,
    "light_bulb": 3,
    "can": 4,
    "glass": 5,
    "ball": 6,
    "marker": 7,
    "cup": 8,
    "remote_control": 9,
}