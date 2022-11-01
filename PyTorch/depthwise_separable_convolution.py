# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/19
    Time: 08:47:36
    
"""
from collections import OrderedDict

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, ReLU6, BatchNorm2d


class DepthSepConv(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(DepthSepConv, self).__init__()
        self.depthwise_conv = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                            groups=in_channels, bias=False)),
            ('bn', BatchNorm2d(in_channels)),
            ('relu', ReLU6(inplace=True))
        ]))
        self.pointwise_conv = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('bn', BatchNorm2d(out_channels)),
            ('relu', ReLU6(inplace=True))
        ]))

    def forward(self, x) -> Tensor:
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)
        return out


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    inputs = torch.randn(1, 6, 10, 10)
    model = DepthSepConv(6, 5)
    output = model(inputs)
    print(output.shape)
    summary(model)
