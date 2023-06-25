import pytest
import torch

from ptrnets.utils.modules import unnormalize


def test_unnormalize_with_default_args():
    # Create input tensor
    tensor = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    # Perform unnormalization with default arguments
    unnormalized_tensor = unnormalize(tensor)

    print(unnormalized_tensor)

    # Define the expected unnormalized tensor
    expected_tensor = torch.tensor([[[[0.0, 0.5], [1, 1.5]]]])

    # Compare the unnormalized tensor with the expected tensor
    assert torch.allclose(unnormalized_tensor, expected_tensor)


def test_unnormalize_with_custom_args():
    # Create input tensor
    tensor = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    # Define custom mean and std
    mean = (0.5,)
    std = (2.0,)

    # Perform unnormalization with custom arguments
    unnormalized_tensor = unnormalize(tensor, mean=mean, std=std)

    # Define the expected unnormalized tensor
    expected_tensor = torch.tensor([[[[0.5, 1.5], [2.5, 3.5]]]])

    # Compare the unnormalized tensor with the expected tensor
    assert torch.allclose(unnormalized_tensor, expected_tensor)


def test_unnormalize_with_inplace_true():
    # Create input tensor
    tensor = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    # Perform unnormalization with inplace=True
    unnormalized_tensor = unnormalize(tensor, inplace=True)

    # Define the expected unnormalized tensor
    expected_tensor = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    # Compare the unnormalized tensor with the expected tensor
    assert torch.allclose(unnormalized_tensor, expected_tensor)
    assert torch.allclose(
        tensor, expected_tensor
    )  # Check if the input tensor is modified


def test_unnormalize_with_invalid_tensor():
    # Create invalid input tensor (not a torch tensor)
    tensor = [1, 2, 3]

    # Check if TypeError is raised
    with pytest.raises(TypeError):
        unnormalize(tensor)


def test_unnormalize_with_invalid_shape():
    # Create invalid input tensor (incorrect shape)
    tensor = torch.tensor([[[0.0, 0.5], [1.0, 1.5]]])

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        unnormalize(tensor)


def test_unnormalize_with_zero_std():
    # Create input tensor
    tensor = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])

    # Define custom mean and std with zero std
    mean = (0.5,)
    std = (0.0,)

    # Check if ValueError is raised due to zero std
    with pytest.raises(ValueError):
        unnormalize(tensor, mean=mean, std=std)
