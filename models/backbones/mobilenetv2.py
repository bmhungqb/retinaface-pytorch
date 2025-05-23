import torch
from torch import nn, Tensor
from torchvision.models import MobileNet_V2_Weights

from models.common import _make_divisible, Conv2dNormActivation, IntermediateLayerGetterByIndex

from typing import Any, List, Optional

__all__ = ["mobilenet_v2", "mobilenet_v2_025", "mobilenet_v2_0125"]


class InvertedResidual(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(
                    in_planes,
                    hidden_dim,
                    kernel_size=1,
                    activation_layer=nn.ReLU6
                )
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_planes
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
        name: str = "mobilenet_v2"
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            dropout (float): The droupout probability
            name (str): Model variant name
        """
        super().__init__()

        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(0.0, width_mult), round_nearest)
        print("input_channel: ", input_channel, "last_channel: ", self.last_channel)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1,  activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mobilenet_v2(*, pretrained: bool = True, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture with width_mult=1.0
    
    Args:
        pretrained (bool): Whether to load pretrained ImageNet weights
        progress (bool): Whether to display download progress for weights
        **kwargs (Any): Additional arguments to pass to the model
    """
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = MobileNetV2(width_mult=1.0, name="mobilenet_v2", **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        model.load_state_dict(state_dict)
    
    param_count = model.count_parameters()
    print(f"MobileNetV2 parameter count: {param_count}")
    
    return model


def mobilenet_v2_025(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture with width_mult=0.25
    
    Args:
        pretrained (bool): Not available for this variant
        progress (bool): Whether to display download progress for weights
        **kwargs (Any): Additional arguments to pass to the model
    """
    if pretrained:
        raise ValueError("No pretrained weights available for MobileNetV2-0.25")
    
    model = MobileNetV2(width_mult=0.25, name="mobilenet_v2_025", **kwargs)
    
    param_count = model.count_parameters()
    print(f"MobileNetV2_025 parameter count: {param_count}")
    
    return model

def mobilenet_v2_0125(*, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture with width_mult=0.125
    
    Args:
        pretrained (bool): Not available for this variant
        progress (bool): Whether to display download progress for weights
        **kwargs (Any): Additional arguments to pass to the model
    """
    if pretrained:
        raise ValueError("No pretrained weights available for MobileNetV2-0.125")
    
    model = MobileNetV2(width_mult=0.125, name="mobilenet_v2_0125", **kwargs)
    
    param_count = model.count_parameters()
    print(f"MobileNetV2_0125 parameter count: {param_count}")
    
    return model

if __name__ == "__main__":
    # Test all three model variants
    print("Testing original MobileNetV2:")
    model_orig = mobilenet_v2(pretrained=False)
    param_count_orig = model_orig.count_parameters()
    print(f"MobileNetV2 parameter count: {param_count_orig}")
    
    print("\nTesting MobileNetV2_025:")
    model_025 = mobilenet_v2_025()
    param_count_025 = model_025.count_parameters()
    print(f"MobileNetV2_025 parameter count: {param_count_025}")