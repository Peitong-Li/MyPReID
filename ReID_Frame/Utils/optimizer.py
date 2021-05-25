"""
@author: Lpt
@email: li2820447@163.com
@file: optimizer.py
@time: 2021/4/21 14:51
@desc: manager optimizer
"""

import torch

__all__ = ['init_optim']

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))