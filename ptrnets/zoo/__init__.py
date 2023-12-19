from typing import Any

import torch
import torchvision
from torch import nn

from .cifar10 import *
from .cifar10 import __all__ as __all__cifar10
from .cornet import *
from .cornet import __all__ as __all__cornet
from .robust import *
from .robust import __all__ as __all__robust
from .shape_biased import *
from .shape_biased import __all__ as __all__shape_biased
from .simclr import *
from .simclr import __all__ as __all__simclr
from .taskonomy import *
from .taskonomy import __all__ as __all__taskonomy
from .vgg_original import *
from .vgg_original import __all__ as __all__vgg_original


AVAILABLE_MODELS = (
    __all__cifar10
    + __all__cornet
    + __all__robust
    + __all__shape_biased
    + __all__simclr
    + __all__taskonomy
    + __all__vgg_original
)


def resnet50_untrained(seed: int = 42, **kwargs: Any) -> nn.Module:
    torch.manual_seed(seed)
    return torchvision.models.resnet50(pretrained=False, **kwargs)
