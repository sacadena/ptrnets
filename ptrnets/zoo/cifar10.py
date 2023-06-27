from typing import Any
from typing import Callable

import torch
from torch import nn
from torchvision.models import resnet50

from ptrnets.utils.config import load_state_dict_from_model_name

__all__ = [
    "resnet50_cifar10",
    "resnet50_cifar10_corrupt0",
    "resnet50_cifar10_corrupt0_2",
    "resnet50_cifar10_corrupt0_4",
    "resnet50_cifar10_corrupt0_6",
    "resnet50_cifar10_corrupt0_8",
    "resnet50_cifar10_corrupt1",
]


def _model(
    model_name: str, model_fn: Callable, pretrained: bool, progress: bool, use_data_parallel: bool, **kwargs: Any
) -> nn.Module:
    model = model_fn(pretrained=False, num_classes=10, **kwargs)
    model = torch.nn.DataParallel(model) if use_data_parallel else model

    if pretrained:
        checkpoint = load_state_dict_from_model_name(model_name, progress=progress)

        if use_data_parallel:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            new_dict = dict()
            for k, v in checkpoint["state_dict"].items():
                new_dict.update({k.replace("module.", ""): v})
            model.load_state_dict(new_dict)

    return model


def resnet50_cifar10(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 starting with seed=42, and with data augmentation"""
    return _model("resnet50_cifar10", resnet50, pretrained, progress, use_data_parallel, **kwargs)


def resnet50_cifar10_corrupt0(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 starting with seed=42, and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt0",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_cifar10_corrupt0_2(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 with label corruption prob = 0.2, starting with seed=42,
    and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt0_2",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_cifar10_corrupt0_4(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 with label corruption prob = 0.4, starting with seed=42,
    and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt0_4",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_cifar10_corrupt0_6(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 with label corruption prob = 0.6, starting with seed=42,
    and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt0_6",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_cifar10_corrupt0_8(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 with label corruption prob = 0.8, starting with seed=42,
    and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt0_8",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_cifar10_corrupt1(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> nn.Module:
    r"""Resnet50 trained on Cifar10 with label corruption prob = 1.0, starting with seed=42,
    and without data augmentation"""
    return _model(
        "resnet50_cifar10_corrupt1",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )
