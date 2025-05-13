import torch
from torch import nn, Tensor
from typing import Any, List, Optional

__all__ = ["mobilenet_v4", "mobilenet_v4_025"]

def _make_divisible(ch: int, divisor: int = 8) -> int:
    return int((ch + divisor / 2) // divisor * divisor)

class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
        activation_layer: Optional[nn.Module] = nn.ReLU6,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale

class UIB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int, use_se: bool) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1))
        layers.append(Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        layers.append(Conv2dNormActivation(hidden_dim, out_channels, kernel_size=1, activation_layer=None))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class MobileNetV4(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_channel = _make_divisible(16 * width_mult)
        last_channel = _make_divisible(1280 * width_mult)

        self.features = nn.Sequential(
            Conv2dNormActivation(3, input_channel, stride=2),
            UIB(input_channel, _make_divisible(16 * width_mult), stride=1, expand_ratio=1, use_se=True),
            UIB(_make_divisible(16 * width_mult), _make_divisible(24 * width_mult), stride=2, expand_ratio=6, use_se=False),
            UIB(_make_divisible(24 * width_mult), _make_divisible(24 * width_mult), stride=1, expand_ratio=6, use_se=False),
            UIB(_make_divisible(24 * width_mult), _make_divisible(40 * width_mult), stride=2, expand_ratio=6, use_se=True),
            UIB(_make_divisible(40 * width_mult), _make_divisible(40 * width_mult), stride=1, expand_ratio=6, use_se=True),
            UIB(_make_divisible(40 * width_mult), _make_divisible(80 * width_mult), stride=2, expand_ratio=6, use_se=False),
            UIB(_make_divisible(80 * width_mult), _make_divisible(80 * width_mult), stride=1, expand_ratio=6, use_se=False),
            UIB(_make_divisible(80 * width_mult), _make_divisible(112 * width_mult), stride=1, expand_ratio=6, use_se=True),
            UIB(_make_divisible(112 * width_mult), _make_divisible(112 * width_mult), stride=1, expand_ratio=6, use_se=True),
            UIB(_make_divisible(112 * width_mult), _make_divisible(160 * width_mult), stride=2, expand_ratio=6, use_se=True),
            UIB(_make_divisible(160 * width_mult), _make_divisible(160 * width_mult), stride=1, expand_ratio=6, use_se=True),
            Conv2dNormActivation(_make_divisible(160 * width_mult), last_channel, kernel_size=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def mobilenet_v4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV4:
    model = MobileNetV4(width_mult=1.0, **kwargs)
    if pretrained:
        # Load pretrained weights if available
        pass  # Placeholder for loading weights
    print(f"MobileNetV4 parameter count: {model.count_parameters()}")
    return model

def mobilenet_v4_025(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV4:
    model = MobileNetV4(width_mult=0.25, **kwargs)
    if pretrained:
        raise ValueError("No pretrained weights available for MobileNetV4-0.25")
    print(f"MobileNetV4_025 parameter count: {model.count_parameters()}")
    return model

if __name__ == "__main__":
    print("Testing MobileNetV4:")
    model = mobilenet_v4(pretrained=False)
    print("\nTesting MobileNetV4_025:")
    model_025 = mobilenet_v4_025()
