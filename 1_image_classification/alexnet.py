import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96,
                               kernel_size=11, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, C*H*W]
        x = self.fc_layers(x)
        return self.softmax(x)


if __name__ == '__main__':
    inputs = torch.zeros((16, 3, 227, 227))
    model = AlexNet(in_channels=3, num_classes=10)
    outputs = model(inputs)  # [16, 3, 227, 227] -> [16, 10]
    print(outputs.size())