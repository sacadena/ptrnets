from typing import Callable
from typing import Dict
from typing import Optional

import torch.utils.model_zoo

from .cornet_r import CORnet_R
from .cornet_rt import CORnet_RT
from .cornet_s import CORnet_S
from .cornet_z import CORnet_Z
from ptrnets.utils.config import load_state_dict_from_model_name


MODEL_MAPPING: Dict[str, Callable] = {
    "cornet_r": CORnet_R,
    "cornet_rt": CORnet_RT,
    "cornet_s": CORnet_S,
    "cornet_z": CORnet_Z,
}


def get_model(
    model_name: str,
    pretrained: bool = False,
    map_location: Optional[str] = None,
    **kwargs,
) -> torch.nn.Module:
    model = MODEL_MAPPING[model_name](**kwargs)
    model = torch.nn.DataParallel(model)
    if pretrained:
        ckpt_data = load_state_dict_from_model_name(
            model_name, map_location=map_location, progress=True
        )
        model.load_state_dict(ckpt_data["state_dict"])
    return model


def cornet_z(
    pretrained: bool = False, map_location: Optional[str] = None
) -> torch.nn.Module:
    return get_model("cornet_z", pretrained=pretrained, map_location=map_location)


def cornet_r(
    pretrained: bool = False, map_location: Optional[str] = None, times: int = 5
) -> torch.nn.Module:
    return get_model(
        "cornet_r", pretrained=pretrained, map_location=map_location, times=times
    )


def cornet_rt(
    pretrained: bool = False, map_location: Optional[str] = None, times: int = 5
) -> torch.nn.Module:
    return get_model(
        "cornet_rt", pretrained=pretrained, map_location=map_location, times=times
    )


def cornet_s(
    pretrained: bool = False, map_location: Optional[str] = None
) -> torch.nn.Module:
    return get_model("cornet_s", pretrained=pretrained, map_location=map_location)
