import io
import os
from contextlib import contextmanager
from importlib.resources import Package
from importlib.resources import path
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import tomli
from torch.hub import load_state_dict_from_url

from ptrnets import zoo
from ptrnets.utils.gdrive import load_state_dict_from_google_drive

PathLike = Union[str, os.PathLike]


class TomlConfig:
    def __init__(self, file_path: PathLike) -> None:
        self._file_path = Path(file_path)
        with self._get_stream() as stream:
            self._dict = tomli.load(stream)

    @contextmanager
    def _get_stream(self) -> Iterator[BinaryIO]:
        if self._file_path.is_file():
            with open(self._file_path, mode="rb") as stream:
                yield io.BytesIO(stream.read())
        else:
            raise ValueError(f"Could not find configuration file {self._file_path}")

    def get_dict(self, key: str) -> Dict[str, Any]:
        if key not in self._dict:
            raise ValueError(f"{key} is not among dict keys")
        d = self._dict.get(key, {})
        if not isinstance(d, dict):
            raise ValueError(f"Not a dictionary: {key}")
        return d

    @property
    def available_keys(self) -> List[str]:
        return list(self._dict.keys())


def get_config_file(package: Package) -> Path:
    with path(package, "config.toml") as f:
        return f


def load_state_dict_from_model_name(
    model_name: str,
    progress: bool = True,
    map_location: Optional[str] = None,
    check_hash: bool = False,
) -> Dict[str, Any]:
    config = TomlConfig(file_path=get_config_file(package=zoo))
    urls = config.get_dict("model_weights").get("url", [])
    gdrive_ids = config.get_dict("model_weights").get("gdrive", [])
    map_location = map_location or "cpu"
    if model_name in urls:
        return load_state_dict_from_url(
            urls.get(model_name),
            progress=progress,
            map_location=map_location,
            check_hash=check_hash,
        )
    if model_name in gdrive_ids:
        gid = gdrive_ids.get(model_name)
        return load_state_dict_from_google_drive(
            gid,
            progress=True,
            filename=f"{gid}.pth.tar",
            map_location=map_location,
        )
    raise ValueError(f"Model {model_name} not found in weight config")
