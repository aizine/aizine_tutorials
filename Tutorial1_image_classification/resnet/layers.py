import torch.nn as nn

from . import blocks


class ResidualLayer(nn.Module):

    def __init__(self, num_blocks,
                 in_channels, out_channels, block=blocks.BasicBlock):
        super(ResidualLayer, self).__init__()
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                blocks.conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
        self.first_block = block(
            in_channels, out_channels, downsample=downsample)
        self.blocks = nn.ModuleList(
            block(out_channels, out_channels) for _ in range(num_blocks))

    def forward(self, x):
        out = self.first_block(x)
        for block in self.blocks:
            out = block(out)
        return out
