import torch
import torch.nn as nn
from collections import abc
from typing import Optional, List


class BasicBlock(nn.Module):
    """ResNet BasicBlock (用于ResNet-18/34)"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck (用于ResNet-50/101/152)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 构建扁平化的层列表，便于逐层迭代和划分
        self.layer_list = nn.ModuleList()

        # stem: conv1 + bn1 + relu + maxpool
        self.layer_list.append(nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))

        # layer1-4: 每个residual block作为一个独立层
        self._make_layer_flat(block, 64, layers[0], stride=1)
        self._make_layer_flat(block, 128, layers[1], stride=2)
        self._make_layer_flat(block, 256, layers[2], stride=2)
        self._make_layer_flat(block, 512, layers[3], stride=2)

        # classifier: avgpool + flatten + fc
        self.layer_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layer_list.append(nn.Flatten())
        self.layer_list.append(nn.Linear(512 * block.expansion, num_classes))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_flat(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        self.layer_list.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            self.layer_list.append(block(self.in_channels, out_channels))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layer_list)

    def __iter__(self):
        return ResNetIterator(self.layer_list)

    def __getitem__(self, index):
        try:
            return self.layer_list[index]
        except IndexError:
            raise StopIteration()


class ResNetIterator(abc.Iterator):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self._index = 0

    def __next__(self):
        if self._index >= len(self.layer_list):
            raise StopIteration()
        layer = self.layer_list[self._index]
        self._index += 1
        return layer


def resnet18(input_channels=3, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channels, num_classes)


def resnet50(input_channels=3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channels, num_classes)
