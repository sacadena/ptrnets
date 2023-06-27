# Custom pytorch modules
from typing import Sequence

import torch
from torch import nn


class Unnormalize(nn.Module):
    """
    Helper class for unnormalizing input tensor
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.0,),
        std: Sequence[float] = (1.0,),
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unnormalize(x, self.mean, self.std, self.inplace)


def unnormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.0,),
    std: Sequence[float] = (1.0,),
    inplace: bool = False,
) -> torch.Tensor:
    """Unnormalize a tensor image by first multiplying by std (channel-wise) and then adding the mean (channel-wise)

    Args:
        tensor (Tensor): Tensor image of size (N, C, H, W) to be de-standarized.
        mean (sequence): Sequence of original means for each channel.
        std (sequence): Sequence of original standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalized Tensor image.

    """

    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor should be a torch tensor. Got {type(tensor)}.")

    if tensor.ndimension() != 4:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (N, C, H, W). Got tensor.size() = {tensor.size()}."
        )
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean_tensor = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std_tensor = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if not std_tensor.all():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    if mean_tensor.ndim == 1:
        mean_tensor = mean_tensor[None, :, None, None]
    if std_tensor.ndim == 1:
        std_tensor = std_tensor[None, :, None, None]

    tensor.mul_(std_tensor).add_(mean_tensor)
    return tensor
