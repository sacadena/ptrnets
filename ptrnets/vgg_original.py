import torch
from torch import nn
import os
from torchvision.models import vgg19, vgg16
from .utils import load_state_dict_from_google_drive
from torch.hub import load_state_dict_from_url

#__all__ = ['vgg19_original', 'vgg19_norm']


google_drive_ids = {
    'vgg19_original': '18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG',
    'vgg19_norm'    : '1r2MAofFyBy3TyazQ7NQOpoAr1dDEAKzL',
}

model_urls = {
    'vgg19_original': '',
    'vgg19_norm'    : '',
}    
    

# Define vgg auxiliary class
class VGGConv(nn.Module):
    def __init__(self):
        super().__init__()
        _vgg19m = vgg19()
        self.features = _vgg19m.features
        self.avgpool  = _vgg19m.avgpool


def _vgg19conv(arch, pretrained, progress, **kwargs):
    model = VGGConv()
    if pretrained:
        try:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        except:
            state_dict = load_state_dict_from_google_drive(google_drive_ids[arch],
                                                  progress=progress, **kwargs)
        model.features.load_state_dict(state_dict)
    return model
    

def vgg19_original(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS
    The weights of this network are those of the original publication and not those from the network trained in pytorch
    Args:
        pretrained (bool): If True, returns the convlayers pre-trained on ImageNet (original weights)
    """
    return _vgg19conv('vgg19_original', pretrained, progress, **kwargs)
    

def vgg19_norm(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>` ONLY CONVOLUTIONAL LAYERS 
    The weights of this network are those of the original publication normalized so that outputs have mean and std equal to one over ImageNet
    Args:
        pretrained (bool): If True, returns the conv layers pre-trained on ImageNet (original weights)
    """
    return _vgg19conv('vgg19_norm', pretrained, progress, **kwargs)

