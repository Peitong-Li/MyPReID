"""
@author: Lpt
@email: li2820447@163.com
@file: __init__.py.py
@time: 2021/4/21 13:45
@desc: 
"""


from __future__ import absolute_import

from .ResNet import *
# from .DenseNet import *
# from .ShuffleNet import *
# from .InceptionV4 import *

__factory = {
    'resnet50': ResNet50,
    # 'resnet101': ResNet101,
    # 'densenet121': DenseNet121,
    # 'shufflenet': ShuffleNet,
    # 'inceptionv4': InceptionV4ReID,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)