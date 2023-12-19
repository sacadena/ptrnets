import unittest
from copy import deepcopy

import torch

from ptrnets import simclr_resnet50x1
from ptrnets import vgg19_norm
from ptrnets.utils.mlayer import clip_model
from ptrnets.utils.mlayer import probe_model


class TestMLayer(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(1, 3, 224, 224)
        self.resnet_model = simclr_resnet50x1(pretrained=False)
        self.vgg_model = vgg19_norm(pretrained=False)

    def test_probe_model(self):
        layer_name = "layer2.1"
        probed_model = probe_model(self.resnet_model, layer_name)
        output = probed_model(self.x)
        self.assertIsNotNone(output)

    def test_clip_model(self):
        layer_name = "features.18"
        clipped_model = clip_model(self.vgg_model, layer_name)
        output = clipped_model(self.x)
        self.assertIsNotNone(output)

    def test_clip_model_is_equal_to_probing(self):
        layer_name = "features.18"
        clipped_model = clip_model(self.vgg_model, layer_name)
        probed_model = probe_model(deepcopy(self.vgg_model), layer_name)
        clipped_output = clipped_model(self.x)
        probed_output = probed_model(self.x)
        self.assertTrue(torch.allclose(clipped_output, probed_output))


if __name__ == "__main__":
    unittest.main()
