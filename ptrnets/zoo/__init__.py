from typing import Any

import torch
import torchvision
from torch import nn

from .cifar10 import *
from .cornet import *
from .robust import *
from .shape_biased import *
from .simclr import *
from .taskonomy import *
from .vgg_original import *


def resnet50_untrained(seed: int = 42, **kwargs: Any) -> nn.Module:
    torch.manual_seed(seed)
    return torchvision.models.resnet50(pretrained=False, **kwargs)
