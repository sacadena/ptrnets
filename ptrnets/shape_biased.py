import os
import inspect
import torch
from torchvision.models import vgg16
from os.path import join, isfile

parent_dir = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
PATH_WEIGHTS = join(parent_dir, 'weights')

filename = join(PATH_WEIGHTS, 'vgg16_train_60_epochs_lr0.01.pth.tar')

def vgg16_shape(pretrained=False):
    return loadshapevgg16(pretrained, filename)

def loadshapevgg16(pretrained, filename):
    'Load Shape-biased Vgg16 network created by Robert Geirhos 2019'

    model = vgg16(pretrained=False)

    if not (pretrained):
        return model
    else:
        if isfile(filename):
            # print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            new_dict = dict()
            for k, v in checkpoint['state_dict'].items():
                new_dict.update({k.replace(".module", ""): v})
            model.load_state_dict(new_dict)
            # print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(filename, checkpoint['epoch']))
        return model
