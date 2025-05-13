import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import List
from models.common import _make_divisible
__all__ = ["mobilenet_v3_025", "mobilenet_v3_0125", "mobilenet_v3"]
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        # print(self.se(x))
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, int(16 * width_mult), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(16 * width_mult))
        self.hs1 = hswish()

        self.bneck1 = nn.Sequential(
            Block(3, int(16 * width_mult), int(16 * width_mult), int(16 * width_mult), nn.ReLU(inplace=True), None, 1),
            Block(3, int(16 * width_mult), int(64 * width_mult), int(32 * width_mult), nn.ReLU(inplace=True), None, 2),
            Block(3, int(32 * width_mult), int(72 * width_mult), int(32 * width_mult), nn.ReLU(inplace=True), None, 1),
            Block(5, int(32 * width_mult), int(72 * width_mult), int(64 * width_mult), nn.ReLU(inplace=True), SeModule(int(64 * width_mult)), 2),
            Block(5, int(64 * width_mult), int(120 * width_mult), int(64 * width_mult), nn.ReLU(inplace=True), SeModule(int(64 * width_mult)), 1),
            Block(5, int(64 * width_mult), int(120 * width_mult), int(64 * width_mult), nn.ReLU(inplace=True), SeModule(int(64 * width_mult)), 1),
        )
        self.bneck2 = nn.Sequential(
            Block(3, int(64 * width_mult), int(240 * width_mult), int(80 * width_mult), hswish(), None, 2),
            Block(3, int(80 * width_mult), int(200 * width_mult), int(80 * width_mult), hswish(), None, 1),
            Block(3, int(80 * width_mult), int(184 * width_mult), int(80 * width_mult), hswish(), None, 1),
            Block(3, int(80 * width_mult), int(184 * width_mult), int(128 * width_mult), hswish(), None, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(3, int(128 * width_mult), int(480 * width_mult), int(128 * width_mult), hswish(), SeModule(int(128 * width_mult)), 1),
            Block(3, int(128 * width_mult), int(672 * width_mult), int(128 * width_mult), hswish(), SeModule(int(128 * width_mult)), 1),
            Block(5, int(128 * width_mult), int(672 * width_mult), int(256 * width_mult), hswish(), SeModule(int(256 * width_mult)), 1),
            Block(5, int(256 * width_mult), int(672 * width_mult), int(256 * width_mult), hswish(), SeModule(int(256 * width_mult)), 2),
            Block(5, int(256 * width_mult), int(960 * width_mult), int(256 * width_mult), hswish(), SeModule(int(256 * width_mult)), 1),
        )

        self.conv2 = nn.Conv2d(int(256 * width_mult), int(960 * width_mult), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(960 * width_mult))
        self.hs2 = hswish()
        self.linear3 = nn.Linear(int(960 * width_mult), int(1280 * width_mult))
        self.bn3 = nn.BatchNorm1d(int(1280 * width_mult))
        self.hs3 = hswish()
        self.linear4 = nn.Linear(int(1280 * width_mult), num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck1(out)
        out = self.bneck2(out)
        out = self.bneck3(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def mobilenet_v3_025(pretrained: bool = False, num_classes: int = 1000):
    model = MobileNetV3(num_classes=num_classes, width_mult=0.25)
    if pretrained:
        # Load pre-trained weights here
        pass
    param_count = model.count_parameters()
    print(f"MobileNetV025 parameter count: {param_count}")
    return model

def mobilenet_v3_0125(pretrained: bool = False, num_classes: int = 1000):
    model = MobileNetV3(num_classes=num_classes, width_mult=0.125)
    if pretrained:
        # Load pre-trained weights here
        pass
    param_count = model.count_parameters()
    print(f"MobileNetV3_0125 parameter count: {param_count}")
    return model

def mobilenet_v3(pretrained: bool = False, num_classes: int = 1000):
    model = MobileNetV3(num_classes=num_classes, width_mult=1.0)
    if pretrained:
        # Load pre-trained weights here
        pass
    param_count = model.count_parameters()
    print(f"MobileNetV3 parameter count: {param_count}")
    return model
