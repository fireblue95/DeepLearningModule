
from collections import OrderedDict
from typing import Dict

from torch import Tensor
from torch.nn import (AvgPool2d, BatchNorm2d, Conv2d, Flatten, Linear, Module,
                      ReLU6, Sequential)


class BottleNeck(Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, t_ratio: int = 1):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.in_c, self.out_c = in_c, out_c

        self.conv = Sequential(OrderedDict([
            ('conv1_pw', Conv2d(in_c, in_c * t_ratio,
                                kernel_size=1, stride=1, bias=False)),
            ('bn1', BatchNorm2d(in_c * t_ratio)),
            ('relu6_1', ReLU6(inplace=True)),

            ('conv2_dw', Conv2d(in_c * t_ratio, in_c * t_ratio, kernel_size=3,
                                stride=stride, padding=1, bias=False, groups=in_c * t_ratio)),
            ('bn2', BatchNorm2d(in_c * t_ratio)),
            ('relu6_2', ReLU6(inplace=True)),

            ('conv3_pw', Conv2d(in_c * t_ratio, out_c,
                                kernel_size=1, stride=1, bias=False)),
            ('bn3', BatchNorm2d(out_c)),
        ]))

        if stride == 1 and in_c != out_c:
            self.short_cut = Sequential(OrderedDict([
                ('conv', Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)),
                ('bn', BatchNorm2d(out_c))
            ]))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.stride == 1:
            if self.in_c != self.out_c:
                out = torch.add(self.short_cut(x), out)
            else:
                out = torch.add(x, out)
        return out


class MobileNetV2(Module):
    def __init__(self, num_classes: int = 80):
        super(MobileNetV2, self).__init__()
        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.layer1 = Sequential(OrderedDict([
            ('conv1', Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', BatchNorm2d(32)),
            ('relu6_1', ReLU6(inplace=True))
        ]))

        self.layer_else = []

        for i, (t, c, n, s) in enumerate(bottleneck_params_list):
            if i == 0:
                self.layer_else.append(self.create_layer(32, t, c, n, s))
            else:
                self.layer_else.append(self.create_layer(
                    bottleneck_params_list[i - 1][1], t, c, n, s))

        self.layer_last = Sequential(OrderedDict([
            ('conv1', Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn1', BatchNorm2d(1280)),
            ('relu6_1', ReLU6(inplace=True)),
            ('avgpool', AvgPool2d(kernel_size=7)),
            ('flatten', Flatten()),
            ('linear', Linear(1280, num_classes))
        ]))

    def create_layer(self, in_c: int, t_ratio: int, out_c: int, repeat: int, stride: int = 1) -> Sequential:
        layer_list = []
        for i in range(repeat):
            layer_name = f'bottleneck' if repeat == 1 else f'bottleneck_{i + 1}'
            if i == 0:
                layer_list.append((layer_name, BottleNeck(
                    in_c, out_c, stride=stride, t_ratio=t_ratio)))
            else:
                layer_list.append((layer_name, BottleNeck(
                    out_c, out_c, stride=1, t_ratio=t_ratio)))

        return Sequential(OrderedDict(layer_list))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)

        for layer in self.layer_else:
            x = layer(x)
        out = self.layer_last(x)

        return out


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    # a = torch.randn(1, 3, 300, 300)
    a = torch.randn(1, 3, 224,  224)
    # model = BottleNeck(3, 64, 1, 6)
    model = MobileNetV2()

    b = model(a)

    print(b.shape)

    summary(model, input_size=(1, 3, 224, 224), device="cpu")
