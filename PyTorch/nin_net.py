# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/22
    Time: 11:21:20
    
"""
from collections import OrderedDict

from torch import Tensor
from torch.nn import Module, Conv2d, ReLU, Sequential, Dropout, MaxPool2d, AdaptiveAvgPool2d, Flatten


class NinBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0) -> None:
        super(NinBlock, self).__init__()
        self.layer = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            ('relu2', ReLU(inplace=True)),
            ('conv3', Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            ('relu3', ReLU(inplace=True))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        return x


class NinNet(Module):
    def __init__(self, num_classes: int) -> None:
        super(NinNet, self).__init__()
        self.num_classes = num_classes
        self.layer = Sequential(OrderedDict([
            ('nin_block1', NinBlock(3, 96, kernel_size=11, stride=4, padding=2)),
            ('drop1', Dropout(p=0.5)),
            ('m_pool1', MaxPool2d(kernel_size=3, stride=2)),
            ('nin_block2', NinBlock(96, 256, kernel_size=5, stride=1, padding=2)),
            ('drop2', Dropout(p=0.5)),
            ('m_pool2', MaxPool2d(kernel_size=3, stride=2)),
            ('nin_block3', NinBlock(256, 384, kernel_size=3, stride=1, padding=1)),
            ('drop3', Dropout(p=0.5)),
            ('m_pool3', MaxPool2d(kernel_size=3, stride=2)),
            ('nin_block4', NinBlock(384, num_classes, kernel_size=3, stride=1, padding=1)),
            ('avg_pool', AdaptiveAvgPool2d(output_size=1)),
            ('flatten', Flatten(start_dim=1))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        return x


if __name__ == '__main__':
    import torch
    from torchinfo import summary
    inputs = torch.randn(1, 3, 224, 224)
    model = NinNet(2)
    out = model(inputs)
    print(out.shape)
    summary(model)