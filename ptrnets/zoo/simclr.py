##################
# SimCLR networks from Chen et al 2020 in pytorch
#
# Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual
# representations. arXiv preprint arXiv:2002.05709.
#
# The original simclr weights were converted ported from tensorflow to pytorch using this repo.
# https://github.com/tonylins/simclr-converter.git
#
# Large portions of the code are an adaptation from https://github.com/tonylins/simclr-converter.git
#
# The original networks were trained on unnormalized data (without normalizing with ImageNet mean and std
# per color channel), in order to ease the use of these networks in comparable settings as the pytorch model zoo,
# the models defined here will scale the inputs with non-trainable parameters by default. (You can remove that
# by setting normalized_inputs=False)
#
# Example usage:
#
# from ptrnets import simclr_resnet50_x1
# model = simclr_resnet50_x1(pretrained=True)
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from ..utils.modules import Unnormalize
from ptrnets.utils.config import load_state_dict_from_model_name


__all__ = ["simclr_resnet50x1", "simclr_resnet50x2", "simclr_resnet50x4"]


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Callable] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Callable] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.downsample = (
            downsample  # hack: moving downsample to the first to make order correct
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ResNet(nn.Module):
    def __init__(
        self,
        block: Any,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable] = None,
        width_mult: int = 1,
        normalized_inputs: bool = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 * width_mult
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.normalized_inputs = normalized_inputs

        if self.normalized_inputs:
            self.unnormalize = Unnormalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(
            block,
            128 * width_mult,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256 * width_mult,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512 * width_mult,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * width_mult, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Any,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]

        if self.normalized_inputs:
            x = self.unnormalize(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _model(
    model_name: str,
    pretrained: bool,
    block: Any,
    layers: List[int],
    width_mult: int,
    normalized_inputs: bool,
    use_data_parallel: bool,
    progress: bool = True,
) -> ResNet:
    model = ResNet(
        block=block,
        layers=layers,
        width_mult=width_mult,
        normalized_inputs=normalized_inputs,
    )

    if pretrained:
        checkpoint = load_state_dict_from_model_name(model_name, progress=progress)

        model.load_state_dict(checkpoint["state_dict"])

        if use_data_parallel:
            model = torch.nn.DataParallel(model)

    return model


def simclr_resnet50x1(
    pretrained: bool = False,
    normalized_inputs: bool = True,
    use_data_parallel: bool = False,
    progress: bool = True,
):
    r"""
    Simclr resnet50 backbone. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    Args:
        pretrained (bool): Pretrain with original weights. Defaults to False
        normalized_inputs (bool): Whether inputs will have zero mean and std one. Defaults to True
        use_data_parallel (bool): Whether to use data parallel for multi GPU. Defaults to False
        progress (bool): Display progress when downloading the model's checkpoint. Defaults to True.
    """
    return _model(
        "simclr-resnet50-x1",
        pretrained=pretrained,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        width_mult=1,
        normalized_inputs=normalized_inputs,
        use_data_parallel=use_data_parallel,
        progress=progress,
    )


def simclr_resnet50x2(
    pretrained: bool = False,
    normalized_inputs: bool = True,
    use_data_parallel: bool = False,
    progress: bool = True,
):
    r"""
    Simclr resnet50 backbone with twice the width. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    Args:
        pretrained (bool): Pretrain with original weights. Defaults to False
        normalized_inputs (bool): Whether inputs will have zero mean and std one. Defaults to True
        use_data_parallel (bool): Whether to use data parallel for multi GPU. Defaults to False
        progress (bool): Display progress when downloading the model's checkpoint. Defaults to True.
    """
    return _model(
        "simclr-resnet50-x2",
        pretrained=pretrained,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        width_mult=2,
        normalized_inputs=normalized_inputs,
        use_data_parallel=use_data_parallel,
        progress=progress,
    )


def simclr_resnet50x4(
    pretrained: bool = False,
    normalized_inputs: bool = True,
    use_data_parallel: bool = False,
    progress: bool = True,
):
    r"""
    Simclr resnet50 backbone with four times the width. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    Args:
        pretrained (bool): Pretrain with original weights. Defaults to False
        normalized_inputs (bool): Whether inputs will have zero mean and std one. Defaults to True
        use_data_parallel (bool): Whether to use data parallel for multi GPU. Defaults to False
        progress (bool): Display progress when downloading the model's checkpoint. Defaults to True.
    """
    return _model(
        "simclr-resnet50-x4",
        pretrained=pretrained,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        width_mult=4,
        normalized_inputs=normalized_inputs,
        use_data_parallel=use_data_parallel,
        progress=progress,
    )
