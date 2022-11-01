# -*- encoding: utf-8 -*-
"""
    Project: All_Models
    Date: 2022/09/19
    Time: 22:43:02

    Total params:
        -> 10,360,620 (is_training=True)
        ->  5,989,892 (is_training=False)

    AuxiliaryClassifier 是輔助分類器，只有在訓練時才會用到，用於計算 loss，權重為 0.3：
        loss = real_loss + 0.3 * aux1 + 0.3 * aux2
"""
from collections import OrderedDict
from typing import List, Tuple, Union

from torch import Tensor, cat
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Module, AvgPool2d, Dropout, Linear, Flatten


class GoogleLeNetV1(Module):
    def __init__(self, num_classes: int, is_training: bool = False) -> None:
        super(GoogleLeNetV1, self).__init__()
        self.is_training = is_training
        self.layer1 = Sequential(OrderedDict([
            ('conv', Conv2d(3, 64, kernel_size=7, stride=2, padding=3)),
            ('relu', ReLU(inplace=True)),
            ('m_pool', MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.layer2 = Sequential(OrderedDict([
            ('conv', Conv2d(64, 192, kernel_size=3, stride=1, padding=1)),
            ('relu', ReLU(inplace=True)),
            ('m_pool', MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.layer3 = Sequential(OrderedDict([
            ('inception_3a', InceptionV1(192, 64, [96, 128], [16, 32], 32)),  # out 256
            ('inception_3b', InceptionV1(256, 128, [128, 192], [32, 96], 64)),  # out 480
            ('m_pool', MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        self.layer4a = Sequential(OrderedDict([
            ('inception_4a', InceptionV1(480, 192, [96, 208], [16, 48], 64))  # out 512
        ]))
        if is_training:
            self.auxiliary_classifier1 = AuxiliaryClassifier(512, num_classes)
        self.layer4d = Sequential(OrderedDict([
            ('inception_4b', InceptionV1(512, 160, [112, 224], [24, 64], 64)),  # out 512
            ('inception_4c', InceptionV1(512, 128, [128, 256], [24, 64], 64)),  # out 512
            ('inception_4d', InceptionV1(512, 112, [144, 288], [32, 64], 64)),  # out 528
        ]))
        if is_training:
            self.auxiliary_classifier2 = AuxiliaryClassifier(528, num_classes)
        self.layer4e = Sequential(OrderedDict([
            ('inception_4e', InceptionV1(528, 256, [160, 320], [32, 128], 128)),  # out 832
            ('m_pool', MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        self.layer5 = Sequential(OrderedDict([
            ('inception_5a', InceptionV1(832, 256, [160, 320], [32, 128], 128)),  # out 832
            ('inception_5b', InceptionV1(832, 384, [192, 384], [48, 128], 128)),  # out 1024
            ('avg_pool', AvgPool2d(kernel_size=7, stride=1)),
            ('dropout', Dropout(p=0.4)),
            ('flatten', Flatten(start_dim=1)),
            ('fc', Linear(1024, num_classes))
        ]))

    def forward(self, x) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out1 = self.layer4a(x)
        if self.is_training:
            aux1 = self.auxiliary_classifier1(out1)
        out2 = self.layer4d(out1)
        if self.is_training:
            aux2 = self.auxiliary_classifier2(out2)
        x = self.layer4e(out2)
        out3 = self.layer5(x)
        if self.is_training:
            return aux1, aux2, out3
        return out3


class AuxiliaryClassifier(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(AuxiliaryClassifier, self).__init__()
        self.auxiliary_classifier = Sequential(OrderedDict([
            ('avg_pool', AvgPool2d(kernel_size=5, stride=3)),
            ('conv', Conv2d(in_channels, 128, kernel_size=1, stride=1)),
            ('flatten', Flatten(start_dim=1)),
            ('fc1', Linear(128 * 4 * 4, 1024)),
            ('fc2', Linear(1024, out_channels))
        ]))

    def forward(self, x):
        result = self.auxiliary_classifier(x)
        return result


class InceptionV1(Module):
    def __init__(self, in_channels: int, stage1_out: int,
                 stage2_out: List[int],
                 stage3_out: List[int],
                 stage4_out: int) -> None:
        super(InceptionV1, self).__init__()
        self.stage1 = Sequential(OrderedDict([
            ('conv', Conv2d(in_channels, stage1_out, kernel_size=1, stride=1)),
            ('relu', ReLU(inplace=True))
        ]))
        self.stage2 = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, stage2_out[0], kernel_size=1, stride=1)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(stage2_out[0], stage2_out[1], kernel_size=3, stride=1, padding=1)),
            ('relu2', ReLU(inplace=True))
        ]))
        self.stage3 = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels, stage3_out[0], kernel_size=1, stride=1)),
            ('relu1', ReLU(inplace=True)),
            ('conv2', Conv2d(stage3_out[0], stage3_out[1], kernel_size=5, stride=1, padding=2)),
            ('relu2', ReLU(inplace=True))
        ]))
        self.stage4 = Sequential(OrderedDict([
            ('m_pool', MaxPool2d(kernel_size=3, stride=1, padding=1)),
            ('conv', Conv2d(in_channels, stage4_out, kernel_size=1, stride=1)),
            ('relu', ReLU(inplace=True))
        ]))

    def forward(self, x) -> Tensor:
        """
        `GoogleNet` 從左到右

        :param x:
        :return: output
        """
        out1 = self.stage1(x)
        out2 = self.stage2(x)
        out3 = self.stage3(x)
        out4 = self.stage4(x)
        result = cat([out1, out2, out3, out4], dim=1)
        return result


if __name__ == '__main__':
    import torch
    from torchinfo import summary

    inputs = torch.randn(1, 3, 224, 224)
    train = False
    model = GoogleLeNetV1(20, train)
    if train:
        o1, o2, o3 = model(inputs)
        print(o1.shape, o2.shape, o3.shape)
    else:
        out = model(inputs)
        print(out.shape)
    summary(model)
