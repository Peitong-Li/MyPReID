"""
@author: Lpt
@email: li2820447@163.com
@file: samplers.py
@time: 2021/4/21 21:02
@desc: 
"""

from __future__ import absolute_import
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
        Randomly sample N identities, then for each identity,
        randomly sample K instances, therefore batch size is N*K.

        Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pids, _) in enumerate(data_source):
            self.index_dic[pids].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        random_pids = torch.randperm(self.num_identities)
        choise_instances_res = []
        for i in random_pids:
            pid = self.pids[i]
            instances = self.index_dic[pid]
            replace = False if len(instances) >= self.num_instances else True
            instances = np.random.choice(instances, size=self.num_instances, replace=replace)
            choise_instances_res.extend(instances)
        # len(choise_instances_res) / num_instances = len(pids)
        return iter(choise_instances_res)

    def __len__(self):
        return self.num_identities * self.num_instances


# if __name__ == '__main__':
#
#     dataset = data_manager.init_img_dataset(name='market1501', root=r'E:\WorkSpace\Datasets')
#     sampler = RandomIdentitySampler(dataset.train)
