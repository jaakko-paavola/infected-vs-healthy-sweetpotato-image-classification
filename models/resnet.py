import torch.nn as nn
import torch.nn.functional as F
from models.model_parts import _weight_initialization, ResNetBlock

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4, input_channels=3, in_planes=64, name="resnet18"):
        super(ResNet, self).__init__()
        if name is not None:
            self.name = name

        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.apply(_weight_initialization)
        
        self.features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, *self.layer1, *self.layer2, *self.layer3, *self.layer4, self.avgpool)
        self.classifier = nn.Sequential(self.linear)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return layers

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def resnet18(num_classes=4) -> ResNet:
    _resnet = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], num_classes=num_classes, name="resnet18")
    return _resnet
