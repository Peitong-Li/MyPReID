"""
@author: Lpt
@email: li2820447@163.com
@file: ResNet.py
@time: 2021/4/21 13:45
@desc: The Model Network
"""
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from IPython import embed

# from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d


class ResNet50(nn.Module):

    def __init__(self, num_classes, loss={'softmax', 'metric'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if not self.loss == {'metric'}:
            self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)

        # if training, return person id probability
        if not self.training:
            return f
        # if evaluate, return features and calclate distance with anchor features
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            return f
        elif self.loss == {'softmax', 'metric'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

# if __name__ == '__main__':
#     model = ResNet50(num_classes=751)
#     imgs = torch.Tensor(32, 3, 256, 128)
#     a = model(imgs)
#     embed()
