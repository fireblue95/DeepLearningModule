# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/19
    Time: 10:36:08
    
"""
from collections import OrderedDict

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, AvgPool2d, Linear, Softmax, BatchNorm2d, ReLU6

from depthwise_separable_convolution import DepthSepConv


class MobileNetV1_300(Module):
    def __init__(self, num_classes: int) -> None:
        """
        input size: [1, 3, 300, 300]


        """
        super(MobileNetV1_300, self).__init__()
        self.num_classes = num_classes
        self.mobile_net = self.build_mobile_net()
        self.avg_pool = AvgPool2d(kernel_size=1, stride=10)
        self.fc = Linear(1024, num_classes)
        self.softmax = Softmax(dim=1)

    @staticmethod
    def build_mobile_net() -> Sequential:
        mobile_net = Sequential(OrderedDict([
            ('conv0', Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)),  # out 150
            ('bn', BatchNorm2d(32)),  # out 150
            ('relu', ReLU6(inplace=True)),  # out 150
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
            ('conv12', DepthSepConv(512, 1024, stride=2)),  # out 10
            ('conv13', DepthSepConv(1024, 1024)),  # out 10
        ]))
        return mobile_net

    def forward(self, x) -> Tensor:
        x = self.mobile_net(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    import torch
    inputs = torch.randn(1, 3, 300, 300)
    model = MobileNetV1_300(2)
    output = model(inputs)
    print(output.shape)
    print(model)
