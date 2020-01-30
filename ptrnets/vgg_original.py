import torch
import os
from os.path import join
from torchvision.models import vgg19
from torch import nn
import inspect

parent_dir = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
PATH_WEIGHTS = join(parent_dir, 'weights')


# Define vgg auxiliary class
class VGGConv(nn.Module):
    def __init__(self):
        super().__init__()
        _vgg19m = vgg19()
        self.features = _vgg19m.features
        self.avgpool  = _vgg19m.avgpool


def vgg19_original(pretrained=False):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS
    The weights of this network are those of the original publication and not those from pytorch
    Args:
        pretrained (bool): If True, returns the convlayers pre-trained on ImageNet (original weights)
    """
    model = VGGConv()
    if pretrained:
        file = join(PATH_WEIGHTS, 'vgg19_original_conv.pth')
        state_dict = torch.load(file)
        model.features.load_state_dict(state_dict)
    return model

def vgg19_norm(pretrained=False):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS with Original Normalized weights
    The weights of this network are those of the original publication normalized so that outputs have mean and std equal to one
    Args:
        pretrained (bool): If True, returns the conv layers pre-trained on ImageNet (original weights)
    """

    model = VGGConv()
    if pretrained:
        file = join(PATH_WEIGHTS, 'vgg19_norm_conv.pth')
        state_dict = torch.load(file)
        model.features.load_state_dict(state_dict)
    return model

