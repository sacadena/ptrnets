import torch
import torchvision

from .cifar10 import *
from .cornet import *
from .robust import *
from .shape_biased import *
from .simclr import *
from .taskonomy import *
from .vgg_original import *


def resnet50_untrained(seed=42, **kwargs):
    torch.manual_seed(seed)
    return torchvision.models.resnet50(pretrained=False, **kwargs)
