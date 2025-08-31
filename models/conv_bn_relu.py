import torch
import torch.nn as nn


class UnstructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None):
        self.mask = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # init mask = ones
        self.mask.weight.data = torch.ones(self.mask.weight.size())

    def update(self, new_mask):
        self.mask.weight.data = new_mask

    def apply(self, conv, bn=None):
        # luôn đưa mask về cùng device với conv
        self.mask = self.mask.to(conv.weight.device)
        conv.weight.data = torch.mul(conv.weight, self.mask.weight)


class StructuredMask:
    def __init__(self, mask):
        self.mask = mask  # có thể là tensor hoặc Parameter

    def apply(self, conv, bn=None):
        # luôn đưa mask về cùng device với conv
        self.mask = self.mask.to(conv.weight.device)

        if isinstance(self.mask, torch.Tensor):
            conv.weight.data = torch.mul(conv.weight, self.mask)
        else:
            conv.weight.data = torch.mul(conv.weight, self.mask.weight)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        relu=True,
    ):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU() if relu else nn.Identity()

        # gắn mask mặc định
        self.mask = UnstructuredMask(
            in_planes, planes, kernel_size, stride, padding, bias
        )

    def forward(self, x):
        # áp dụng mask trước khi conv
        self.mask.apply(self.conv, self.bn)
        return self.relu(self.bn(self.conv(x)))
