"""
@author: Lpt
@email: li2820447@163.com
@file: dataset_loader.py
@time: 2021/4/20 21:42
@desc: Make the data out in your way
@refer: https://github.com/michuanhaohao/AlignedReID/
"""

import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_image = False

    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_image:
        try:
            img = Image.open(img_path).convert('RGB')
            got_image = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_path, pid, camid = self.dataset[item]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


# if __name__ == '__main__':
#     import data_manager
#     dataset = data_manager.init_img_dataset(name='market1501', root=r'E:\WorkSpace\Datasets')
#     train_loader = ImageDataset(dataset.train)
#     # from IPython import embed
#     # embed()
#     for batch_id, (img, pid, camid) in enumerate(train_loader):
#         break
#     print(batch_id)
#     img.save('a.jpg')



