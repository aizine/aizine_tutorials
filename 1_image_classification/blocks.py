import torch
import torch.nn as nn


#  kernel_sizeが3x3，padding=stride=1のconvは非常によく使用するので、関数で簡単い呼べるようにする
def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True,
                     dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    #  Implementation of Basic Building Block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity_x = x  # hold input for shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)

        out += identity_x  # shortcut connection
        return self.relu(out)


class BottleneckBlock(nn.Module):
    #  Implementation of Bottleneck Building Block

    def __init__(self, in_channels, hid_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2(hid_channels)
        self.conv2 = conv3x3(hid_channels, hid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.conv3 = conv1x1(hid_channels, in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)

        out += identity_x

        return self.relu(out)
