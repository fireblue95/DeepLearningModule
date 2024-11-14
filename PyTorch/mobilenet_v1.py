
from collections import OrderedDict
from typing import List

import numpy as np
from numpy import ndarray
from torch.nn import (AvgPool2d, BatchNorm2d, Conv2d, Flatten, Linear, Module,
                      ReLU, Sequential)


class DWSepConv(Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, padding=1):
        super(DWSepConv, self).__init__()

        self.depthwise_conv = Sequential(OrderedDict([
            ('conv', Conv2d(in_c, in_c, kernel_size=3,
             stride=stride, padding=padding, groups=in_c, bias=False)),
            ('bn', BatchNorm2d(in_c)),
            ('relu', ReLU(inplace=True)),
        ]))

        self.pointwise_conv = Sequential(OrderedDict([
            ('conv', Conv2d(in_c, out_c, kernel_size=1,
             stride=1, bias=False)),
            ('bn', BatchNorm2d(out_c)),
            ('relu', ReLU(inplace=True)),
        ]))

    def forward(self, x):
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)

        return out


class MobileNetV1(Module):
    def __init__(self, num_classes: int, alpha: float = 1.0):
        """
            alpha: width_multiplier
        """
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes

        # self.mobile_net = Sequential(OrderedDict([
        #     ('conv1', Conv2d(3, 32, kernel_size=3, stride=2, padding=1)),
        #     ('bn', BatchNorm2d(32)),
        #     ('relu', ReLU(inplace=True)),
        #     ('conv2', DWSepConv(32, 64)),
        #     ('conv3', DWSepConv(64, 128, stride=2)),
        #     ('conv4', DWSepConv(128, 128)),
        #     ('conv5', DWSepConv(128, 256, stride=2)),
        #     ('conv6', DWSepConv(256, 256)),
        #     ('conv7', DWSepConv(256, 512, stride=2)),
        #     ('conv8_1', DWSepConv(512, 512)),
        #     ('conv8_2', DWSepConv(512, 512)),
        #     ('conv8_3', DWSepConv(512, 512)),
        #     ('conv8_4', DWSepConv(512, 512)),
        #     ('conv8_5', DWSepConv(512, 512)),
        #     ('conv9', DWSepConv(512, 1024, stride=2)),
        #     ('conv10', DWSepConv(1024, 1024)),
        #     ('avg_pool', AvgPool2d(kernel_size=7, stride=1)),
        #     ('flatten', Flatten(start_dim=1)),
        #     ('fc', Linear(1024, num_classes))
        # ]))

        channels: ndarray = np.array([32, 64, 128, 128,
                                      256, 256,
                                      512, 512, 512, 512, 512, 512,
                                      1024, 1024])

        channels = (channels * alpha).astype(np.int16)

        conv_name = [f'conv{x}' for x in range(2, 8)] + \
            [f'conv8_{x}' for x in range(1, 6)] + \
            [f'conv{x}' for x in range(9, 11)]

        out_list: List = []
        for i in range(len(channels)):
            if i == 0:
                out_list += [('conv1', Conv2d(3, channels[i], kernel_size=3, stride=2, padding=1)),
                             ('bn', BatchNorm2d(channels[i])),
                             ('relu', ReLU(inplace=True))]
            else:
                out_list.append((conv_name[i - 1],
                                 DWSepConv(channels[i - 1], channels[i],
                                           stride=2 if conv_name[i - 1] in ['conv3', 'conv5', 'conv7', 'conv9'] else 1)))

        out_list += [('avg_pool', AvgPool2d(kernel_size=7, stride=1)),
                     ('flatten', Flatten(start_dim=1)),
                     ('fc', Linear(channels[-1], num_classes))]

        self.mobile_net = Sequential(OrderedDict(out_list))

    def forward(self, x):
        x = self.mobile_net(x)
        return x


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    # a = torch.randn(1, 16, 7, 7)
    # model = DWSepConv(16, 32, 2)
    # a = torch.randn(1, 3, 300, 300)
    a = torch.randn(1, 3, 224,  224)
    model = MobileNetV1(1000, alpha=1)

    b = model(a)
    print(b.shape)

    summary(model)
