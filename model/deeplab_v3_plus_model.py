import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_model import *

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepLabV3Plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.head = _DeepLabHead()
        self.decoder1 = nn.Conv2d(64, 48, 1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(304, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        out, branch = self.resnet(x)
        _, _, uh, uw = branch.size()
        out = self.head(out)
        out = F.interpolate(out, size=(uh, uw), mode='bilinear', align_corners=True)
        branch = self.decoder1(branch)
        out = torch.cat([out, branch], 1)
        out = self.decoder2(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

class _DeepLabHead(nn.Module):
    def __init__(self):
        super(_DeepLabHead, self).__init__()
        self.aspp = ASPP(2048, [6, 12, 18])
    def forward(self, x):
        out = self.aspp(x)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        
        self.imagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.conv1 = self._ASPPConv(in_channels, out_channels, rate1)
        self.conv2 = self._ASPPConv(in_channels, out_channels, rate2)
        self.conv3 = self._ASPPConv(in_channels, out_channels, rate3)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        features1 = F.interpolate(self.imagepool(x), size=(h, w), mode='bilinear', align_corners=True)

        features2 = self.conv1x1(x)
        features3 = self.conv1(x)
        features4 = self.conv2(x)
        features5 = self.conv3(x)
        out = torch.cat((features1, features2, features3, features4, features5), 1)
        out = self.project(out)
        return out
    
    def _ASPPConv(self, in_channels, out_channels, atrous_rate):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, padding=atrous_rate,
                    dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        return block


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


