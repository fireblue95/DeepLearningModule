# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/19
    Time: 11:40:52
    
"""
from collections import OrderedDict
from typing import Tuple

from torch import Tensor
from torch.nn import Module, Conv2d, Sequential, ReLU6, BatchNorm2d

from depthwise_separable_convolution import DepthSepConv


class ConvBNReLU(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0,
                 bias: bool = False) -> None:
        super(ConvBNReLU, self).__init__()
        if not bias:  # bias == False
            self.convBnRelu = Sequential(OrderedDict([
                ('conv', Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)),
                ('bn', BatchNorm2d(out_channels)),
                ('relu', ReLU6(inplace=True))
            ]))
        else:  # bias == True
            self.convBnRelu = Sequential(OrderedDict([
                ('conv', Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)),
                ('relu', ReLU6(inplace=True))
            ]))

    def forward(self, x: Tensor) -> Tensor:
        out = self.convBnRelu(x)
        return out


class MobileNetSSD(Module):
    def __init__(self) -> None:
        super(MobileNetSSD, self).__init__()

        self.conv11 = self.build_mobile_net()
        self.conv13 = Sequential(OrderedDict([
            ('conv12', DepthSepConv(512, 1024, stride=2)),  # out 10
            ('conv13', DepthSepConv(1024, 1024)),  # out 10
        ]))
        self.conv14 = Sequential(OrderedDict([
            ('conv14_1', ConvBNReLU(1024, 256, kernel_size=1, stride=1)),  # out 10
            ('conv14_2', ConvBNReLU(256, 512, kernel_size=3, stride=2, padding=1)),  # out 5
        ]))
        self.conv15 = Sequential(OrderedDict([
            ('conv15_1', ConvBNReLU(512, 128, kernel_size=1, stride=1)),  # out 5
            ('conv15_2', ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1)),  # out 3
        ]))
        self.conv16 = Sequential(OrderedDict([
            ('conv16_1', ConvBNReLU(256, 128, kernel_size=1, stride=1)),  # out 3
            ('conv16_2', ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1)),  # out 2
        ]))
        self.conv17 = Sequential(OrderedDict([
            ('conv17_1', ConvBNReLU(256, 64, kernel_size=1, stride=1)),  # out 2
            ('conv17_2', ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1, bias=True)),  # out 1
        ]))

    @staticmethod
    def build_mobile_net() -> Sequential:
        mobile_net = Sequential(OrderedDict([
            ('conv0', ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)),  # out 150
            ('conv1', DepthSepConv(32, 64)),  # out 150
            ('conv2', DepthSepConv(64, 128, stride=2)),  # out 75
            ('conv3', DepthSepConv(128, 128)),  # out 75
            ('conv4', DepthSepConv(128, 256, stride=2)),  # out 38
            ('conv5', DepthSepConv(256, 256)),  # out 38
            ('conv6', DepthSepConv(256, 512, stride=2)),  # out 19
            ('conv7', DepthSepConv(512, 512)),  # out 19
            ('conv8', DepthSepConv(512, 512)),  # out 19
            ('conv9', DepthSepConv(512, 512)),  # out 19
            ('conv10', DepthSepConv(512, 512)),  # out 19
            ('conv11', DepthSepConv(512, 512)),  # out 19
        ]))
        return mobile_net

    def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
        endpoints = dict()
        x = self.conv11(x)
        endpoints['block4'] = x
        x = self.conv13(x)
        endpoints['block7'] = x
        x = self.conv14(x)
        endpoints['block8'] = x
        x = self.conv15(x)
        endpoints['block9'] = x
        x = self.conv16(x)
        endpoints['block10'] = x
        x = self.conv17(x)
        endpoints['block11'] = x
        return x, endpoints


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    model = MobileNetSSD()
    # print(model)
    inputs = torch.randn(1, 3, 300, 300)
    output, endpoints = model(inputs)
    print(output.shape)
    summary(model)

    for k, v in endpoints.items():
        print(k, v.shape)
