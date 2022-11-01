# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/22
    Time: 09:14:32
    
"""
from collections import OrderedDict

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, ReLU, AdaptiveAvgPool2d, Dropout, Linear


class AlexNet(Module):
    def __init__(self, num_classes: int) -> None:
        """
        input: [224, 224, 3]

        """
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.layer = Sequential(OrderedDict([
            ('conv1', Conv2d(3, 96, kernel_size=11, stride=4, padding=2)),
            ('relu1', ReLU(inplace=True)),
            ('m_pool1', MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', Conv2d(96, 256, kernel_size=5, stride=1, padding=2)),
            ('relu2', ReLU(inplace=True)),
            ('m_pool2', MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', Conv2d(256, 384, kernel_size=3, stride=1, padding=1)),
            ('relu3', ReLU(inplace=True)),
            ('conv4', Conv2d(384, 384, kernel_size=3, stride=1, padding=1)),
            ('relu4', ReLU(inplace=True)),
            ('conv5', Conv2d(384, 256, kernel_size=3, stride=1, padding=1)),
            ('relu5', ReLU(inplace=True)),
            ('m_pool3', MaxPool2d(kernel_size=3, stride=2)),
        ]))
        self.avgpool = AdaptiveAvgPool2d(output_size=6)
        self.classifier = Sequential(OrderedDict([
            ('drop1', Dropout(p=0.5)),
            ('fc1', Linear(256 * 6 * 6, 4096)),
            ('relu1', ReLU(inplace=True)),
            ('drop2', Dropout(p=0.5)),
            ('fc2', Linear(4096, 4096)),
            ('relu2', ReLU(inplace=True)),
            ('fc3', Linear(4096, num_classes))
        ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


if __name__ == '__main__':
    import torch

    inputs = torch.randn(1, 3, 224, 224)
    model = AlexNet(2)
    out = model(inputs)
    print(out.shape)
    # from torchvision.models import AlexNet
    #
    # model = AlexNet()
    print(model)
