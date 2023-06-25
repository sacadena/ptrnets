import errno
import os
import re
import sys
import zipfile
from typing import Any
from typing import Dict
from typing import Optional

import gdown
import requests
import torch
from gdown.parse_url import parse_url


def _get_name(id_):
    """
    Gets the base url name from the gdrive using the file id.
    """

    url = f"https://drive.google.com/uc?id={id_}"
    sess = requests.session()
    file_id, is_download_link = parse_url(url)

    while True:
        try:
            res = sess.get(url, stream=True)
        except requests.exceptions.ProxyError as e:
            print("An error has occurred using proxy:", file=sys.stderr)
            print(e, file=sys.stderr)
            return

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (file_id and is_download_link):
            break

        url = res.url  # Update the URL if redirected

    if file_id and is_download_link:
        m = re.search('filename="(.*)"', res.headers["Content-Disposition"])
        output = m.groups()[0]
    else:
        output = os.basename(url)

    return output


def load_state_dict_from_google_drive(
    id_: str,
    model_dir: Optional[str] = None,
    map_location: Optional[str] = None,
    progress: bool = True,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Loads the Torch serialized object at the given google drive file id.
    Note: Inspired by torch.hub.load_state_dict_from_url

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if not set.

    Args:
        id_ (string): GDrive file id
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        filename (string, optional): name of the file to store

    Example:
        >>> state_dict = load_state_dict_from_google_drive(id_='18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG')

    """

    if model_dir is None:
        torch_home = torch.hub._get_torch_home()
        model_dir = os.path.join(torch_home, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    if filename is None:
        filename = _get_name(id_)  # use default name of the file in gdrive

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = f"https://drive.google.com/uc?id={id_}"
        gdown.download(url, cached_file, quiet=not (progress))

    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()

            if len(members) != 1:
                raise RuntimeError("Only one file(not dir) is allowed in the zipfile")

            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)

    print(cached_file)
    map_location = map_location or "cpu"
    return torch.load(cached_file, map_location=map_location)
