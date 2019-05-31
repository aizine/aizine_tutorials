import torch
import torch.nn as nn

from . import layers


class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = layers.ResidualLayer(2, in_channels=64, out_channels=64)
        self.layer2 = layers.ResidualLayer(2, in_channels=64, out_channels=128)
        self.layer3 = layers.ResidualLayer(
            2, in_channels=128, out_channels=256)
        self.layer4 = layers.ResidualLayer(
            2, in_channels=256, out_channels=512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    inputs = torch.zeros((16, 3, 227, 227))
    model = ResNet18(num_classes=10)
    outputs = model(inputs)  # [16, 3, 227, 227] -> [16, 10]
    print(outputs.size())
