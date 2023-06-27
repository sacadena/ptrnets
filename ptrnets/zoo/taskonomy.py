from typing import Any

import torch

from ptrnets.utils.config import load_state_dict_from_model_name
from ptrnets.zoo.third_party.taskonomy.networks import LIST_OF_TASKS
from ptrnets.zoo.third_party.taskonomy.networks import TaskonomyEncoder


encoders = ["_".join([task, "encoder"]) for task in LIST_OF_TASKS]  # names of taskonomy encoder networks

__all__ = encoders


def _model(
    model_name: str, pretrained: bool, progress: bool, use_data_parallel: bool, **kwargs: Any
) -> torch.nn.Module:
    model = TaskonomyEncoder()
    if pretrained:
        checkpoint = load_state_dict_from_model_name(model_name, progress=progress)
        model.load_state_dict(checkpoint["state_dict"])
    return torch.nn.DataParallel(model) if use_data_parallel else model


def autoencoding_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """Autoencoding encoder network"""
    return _model("autoencoding_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def class_object_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("class_object_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def class_scene_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("class_scene_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def colorization_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("colorization_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def curvature_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("curvature_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def denoising_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("denoising_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def depth_euclidean_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "depth_euclidean_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def depth_zbuffer_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("depth_zbuffer_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def edge_occlusion_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "edge_occlusion_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def edge_texture_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("edge_texture_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def egomotion_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("egomotion_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def fixated_pose_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("fixated_pose_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def inpainting_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("inpainting_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def jigsaw_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("jigsaw_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def keypoints2d_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("keypoints2d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def keypoints3d_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("keypoints3d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def nonfixated_pose_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "nonfixated_pose_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def normal_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("normal_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def point_matching_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "point_matching_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def reshading_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("reshading_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def room_layout_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model("room_layout_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs)


def segment_semantic_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "segment_semantic_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def segment_unsup25d_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "segment_unsup25d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def segment_unsup2d_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "segment_unsup2d_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )


def vanishing_point_encoder(
    pretrained: bool = True, progress: bool = True, use_data_parallel: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """encoder network"""
    return _model(
        "vanishing_point_encoder", pretrained, progress=progress, use_data_parallel=use_data_parallel, **kwargs
    )
