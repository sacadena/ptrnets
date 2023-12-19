# ptrnets

[![Release](https://img.shields.io/github/v/release/sacadena/ptrnets)](https://img.shields.io/github/v/release/sacadena/ptrnets)
[![Build status](https://img.shields.io/github/actions/workflow/status/sacadena/ptrnets/main.yml?branch=master)](https://github.com/sacadena/ptrnets/actions/workflows/main.yml?query=branch%3AMain)
[![codecov](https://codecov.io/gh/sacadena/ptrnets/branch/master/graph/badge.svg)](https://codecov.io/gh/sacadena/ptrnets)
[![Commit activity](https://img.shields.io/github/commit-activity/m/sacadena/ptrnets)](https://img.shields.io/github/commit-activity/m/sacadena/ptrnets)
[![License](https://img.shields.io/github/license/sacadena/ptrnets)](https://img.shields.io/github/license/sacadena/ptrnets)

Collection of pretrained networks in pytorch readily available for transfer learning tasks like neural system identification.

## Installation

```bash
pip install ptrnets
```

## Usage
Find a list of all available models like this:

```python
from ptrnets import AVAILABLE_MODELS

print(AVAILABLE_MODELS)
```

Import a model like this:

```python
from ptrnets import simclr_resnet50x2

model = simclr_resnet50x2(pretrained=True)
```
You can access intermediate representations in two ways:

### Probing the model
You can conveniently access intermediate representations of a forward pass using the `ptrnets.utils.mlayer.probe_model` function Example:
```python 
import torch
from ptrnets import resnet50
from ptrnets.utils.mlayer import probe_model

model = resnet50(pretrained=True)
available_layers = [name for name, _ in model.named_modules()]
layer_name = "layer2.1"
assert layer_name in available_layers, f"Layer {layer_name} not available. Choose from {available_layers}"

model_probe = probe_model(model, layer_name)

x = torch.rand(1, 3, 224, 224)
output = model_probe(x)
```

**Note**: if the input is not large enough to do a full forward pass through the network, you might need to use a `try-except` block to catch the `RuntimeError`.

### Clipping the model

`ptrnets.utils.mlayer.clip_model` creates a copy of the model up to a specific layer. Because the model is smaller, a forward pass can run faster. 
However, the output is only guaranteed to be the same as the original model's if the architecture is fully sequential up until that layer. 

Example:
```python
import torch
from ptrnets import vgg16
from ptrnets.utils.mlayer import clip_model, probe_model

model = vgg16(pretrained=True)
available_layers = [name for name, _ in model.named_modules()]
layer_name = "features.18"
assert layer_name in available_layers, f"Layer {layer_name} not available. Choose from {available_layers}"

model_clipped = clip_model(model, layer_name)  # Creates new model up to the layer

x = torch.rand(1, 3, 224, 224)
output = model_clipped(x)

assert torch.allclose(output, probe_model(model, layer_name)(x)), "Output of clipped model is not the same as the original model"
```

## Contributing
Pull requests are welcome. 
Please see instructions [here](https://github.com/sacadena/ptrnets/blob/master/CONTRIBUTING.rst).




