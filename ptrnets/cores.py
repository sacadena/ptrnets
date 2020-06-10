from collections import OrderedDict, Iterable
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from .utils import clip_model


class Core:
    def initialize(self):
        raise NotImplementedError("Not initializing")

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x or "skip" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda):
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
                

class TaskCore(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        model_name,
        layer_name,
        pretrained=True,
        bias=False,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs
    ):
        """
        Core from pretrained networks on image tasks.

        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            model_name (str): Name of the image recognition task model. Possible are all models in
            ptrnets: torchvision.models plus others
            layer_name (str): Name of the layer at which to clip the model
            pretrained (boolean): Whether to use a randomly initialized or pretrained network (default: True)
            bias (boolean): Whether to keep bias weights in the output layer (default: False)
            final_batchnorm (boolean): Whether to add a batch norm after the final conv layer (default: True)
            final_nonlinearity (boolean): Whether to add a final nonlinearity (ReLU) (default: True)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to freeze gradients of the core or to allow training
        """
        if kwargs:
            warnings.warn(
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__), UserWarning
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        model = getattr(ptrnets, model_name)(pretrained=pretrained)
        model_clipped = clip_model(model, layer_name)
        
        # Remove the bias of the last conv layer if not :bias:
        if not bias:
            if 'bias' in model_clipped[-1]._parameters:
                zeros = torch.zeros_like(model_clipped[-1].bias)
                model_clipped[-1].bias.data = zeros
        
        # Fix pretrained parameters during training
        if not fine_tune:
            for param in model_clipped.parameters():
                param.requires_grad = False

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module("TaskDriven", model_clipped)
        
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, input_):
        # If model is designed for RBG input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1 and self.features.TaskDriven[0].in_channels == 3:
            input_ = input_.repeat(1, 3, 1, 1)
        input_ = self.features(input_)
        return input_

    def regularizer(self):
        return 0

    @property
    def outchannels(self):
        """
        Function which returns the number of channels in the output conv layer. If the output layer is not a conv
        layer, the last conv layer in the network is used.

        Returns: Number of output channels
        """
        found_outchannels = False
        i = 1
        while not found_outchannels:
            if "out_channels" in self.features.TaskDriven[-i].__dict__:
                found_outchannels = True
            else:
                i += 1
        return self.features.TaskDriven[-i].out_channels

    def initialize(self, cuda=False):
        # Overwrite parent class's initialize function because initialization is done by the 'pretrained' parameter
        self.put_to_cuda(cuda=cuda)