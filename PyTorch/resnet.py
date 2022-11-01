# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/19
    Time: 07:57:40

    Total params: num_classes=20
        ResNet  18 -> 11,012,948
        ResNet  34 -> 21,121,108
        ResNet  50 -> 23,549,012
        ResNet 101 -> 42,541,140
        ResNet 152 -> 58,184,788
"""
from collections import OrderedDict
from typing import List

from torch import Tensor, add
from torch.nn import Module, Sequential, Conv2d, ReLU, BatchNorm2d, MaxPool2d, AvgPool2d, Flatten, Linear


class BasicBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.stage = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', BatchNorm2d(out_channels)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', BatchNorm2d(out_channels))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage(x)
        return x


class ConvBlock(Module):
    def __init__(self, in_channels: int, filters: List[int], stride: int = 2) -> None:
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, F1, kernel_size=1, stride=stride, bias=False)),
            # in[N, 64, 56, 56] out[N, 64, 56, 56]
            ('bn1', BatchNorm2d(F1)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(F1, F2, kernel_size=3, stride=1, padding=1, bias=False)),
            # in[N, 64, 56, 56], out[N, 64, 56+2-2-1/1+1]
            ('bn2', BatchNorm2d(F2)),
            ('relu2', ReLU(inplace=True)),
            ('conv3', Conv2d(F2, F3, kernel_size=1, stride=1, bias=False)),
            ('bn3', BatchNorm2d(F3))
        ]))
        self.shortcut_1 = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels, F3, kernel_size=1, stride=stride, bias=False)),
            ('bn', BatchNorm2d(F3))
        ]))
        self.relu = ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.stage(x)
        x_shortcut = self.shortcut_1(x)
        y = add(y, x_shortcut)
        y = self.relu(y)
        return y


class IdentityBlock(Module):
    def __init__(self, in_channels: int, filters: List[int]) -> None:
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, F1, kernel_size=1, stride=1, bias=False)),
            ('bn1', BatchNorm2d(F1)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(F1, F2, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', BatchNorm2d(F2)),
            ('relu2', ReLU(inplace=True)),
            ('conv3', Conv2d(F2, F3, kernel_size=1, stride=1, bias=False)),
            ('bn3', BatchNorm2d(F3))
        ]))
        self.relu = ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.stage(x)
        y = add(y, x)
        y = self.relu(y)
        return y


class ResNet(Module):
    def __init__(self, num_classes: int, layer_num: int) -> None:
        super(ResNet, self).__init__()
        layer_count = [18, 34, 50, 101, 152]  # choice net constructures
        assert layer_num in layer_count, f"layer_num is not in {layer_count}"
        # every net parameters
        every_block_num = {18: [2, 2, 2, 2],
                           34: [3, 4, 6, 3],
                           50: [3, 4, 6, 3],
                           101: [3, 4, 23, 3],
                           152: [3, 8, 36, 3]}
        block_num = every_block_num[layer_num]

        self.layer1 = Sequential(OrderedDict([
            ('conv', Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),  # 224 -> 112
            ('bn', BatchNorm2d(64)),
            ('relu', ReLU(inplace=True)),
            ('m_pool', MaxPool2d(kernel_size=3, stride=2, padding=1)),  # 112 -> 56
        ]))

        if layer_num in [18, 34]:
            # ResNet18 34 vvv----------
            self.layer2 = Sequential(OrderedDict([
                ('basic_block1', BasicBlock(64, 64))] +
                [(f'basic_block{x + 1}', BasicBlock(64, 64)) for x in range(1, block_num[0])]
            ))
            self.layer3 = Sequential(OrderedDict([
                ('basic_block1', BasicBlock(64, 128, stride=2))] +
                [(f'basic_block{x + 1}', BasicBlock(128, 128)) for x in range(1, block_num[1])]
            ))
            self.layer4 = Sequential(OrderedDict([
                ('basic_block1', BasicBlock(128, 256, stride=2))] +
                [(f'basic_block{x + 1}', BasicBlock(256, 256)) for x in range(1, block_num[2])]
            ))
            self.layer5 = Sequential(OrderedDict([
                ('basic_block1', BasicBlock(256, 512, stride=2))] +
                [(f'basic_block{x + 1}', BasicBlock(512, 512)) for x in range(1, block_num[3])]
            ))
            out_channels = 512
            # ResNet18 34 ^^^----------
        else:
            self.layer2 = Sequential(OrderedDict([
                ('conv_block', ConvBlock(64, [64, 64, 256], stride=1))] +
                [(f'identity_block{x}', IdentityBlock(256, [64, 64, 256])) for x in range(1, block_num[0])]
            ))
            self.layer3 = Sequential(OrderedDict([
                ('conv_block', ConvBlock(256, [128, 128, 512]))] +
                [(f'identity_block{x}', IdentityBlock(512, [128, 128, 512])) for x in range(1, block_num[1])]
            ))
            self.layer4 = Sequential(OrderedDict([
                ('conv_block', ConvBlock(512, [256, 256, 1024]))] +
                [(f'identity_block{x}', IdentityBlock(1024, [256, 256, 1024])) for x in range(1, block_num[2])]
            ))
            self.layer5 = Sequential(OrderedDict([
                ('conv_block', ConvBlock(1024, [512, 512, 2048]))] +
                [(f'identity_block{x}', IdentityBlock(2048, [512, 512, 2048])) for x in range(1, block_num[3])]
            ))
            out_channels = 2048
        self.layer_final = Sequential(OrderedDict([
            ('avg_pool', AvgPool2d(kernel_size=7, stride=1)),
            ('flatten', Flatten(start_dim=1)),
            ('fc', Linear(out_channels, num_classes))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_final(x)
        return x


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    inputs = torch.randn(1, 3, 224, 224)
    model = ResNet(num_classes=20, layer_num=101)
    out = model(inputs)
    print(out.shape)
    # print(model)
    summary(model)
