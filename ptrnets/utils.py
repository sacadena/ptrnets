import requests
import torch
import os
import sys
import hashlib
import re
import errno
from tqdm import tqdm
import warnings
import zipfile
import gdown
from gdown.parse_url import parse_url
from torch import nn


def clip_model(model, layer_name):
    """
    Returns a copy of the model up to :layer_name:
    Params:
        model: (nn.Module instance)
        layer_name (string) Should be among the named modules of model
    Returns:
        clipped_model: (nn.Sequential) copy of model up to the layer
    Examples:
        from torchvision.models import resnet50, vgg19
        resnet = resnet50(pretrained=True)
        vgg    = vgg19(pretrained=True)
        
        clipped_resnet = clip_model(resnet, 'layer3.0.conv2')
        clipped_vgg    = clip_model(vgg, 'features.10')
    """
    
    assert layer_name in [n for n,_ in model.named_modules()], 'No module named {}'.format(layer_name)
    
    features = OrderedDict()
    nodes_iter = iter(layer_name.split('.'))
    
    def recursive(module, node = next(nodes_iter), prefix=[]):
        
        for name, layer in module.named_children():
            fullname = ".".join(prefix+[name])
            if (name == node) and (fullname != layer_name):
                recursive(layer, node = next(nodes_iter), prefix=[fullname])
                return
            else:
                features[name] = layer
                #print(fullname)
                if fullname == layer_name:
                    return

    recursive(model)
    
    clipped_model = nn.Sequential(features)
    
    return clipped_model



def _get_name(id):
    """
    Gets the base url name from the gdrive using the file id.
    """
    
    url = 'https://drive.google.com/uc?id={}'.format(id)
    
    url_origin = url
    sess = requests.session()

    file_id, is_download_link = parse_url(url)
    
    while True:
        try:
            res = sess.get(url, stream=True)
        except requests.exceptions.ProxyError as e:
            print("An error has occurred using proxy:", proxy, file=sys.stderr)
            print(e, file=sys.stderr)
            return

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (file_id and is_download_link):
            break
    
    if file_id and is_download_link:
        m = re.search(
            'filename="(.*)"', res.headers["Content-Disposition"]
        )
        output = m.groups()[0]
    else:
        output = osp.basename(url)
    
    return output



def load_state_dict_from_google_drive(id, model_dir=None, map_location=None, progress=True, check_hash=False, filename=None):
    r"""Loads the Torch serialized object at the given google drive file id.
    Note: Inspired by torch.hub.load_state_dict_from_url

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        filename (string, optional): name of the file to store

    Example:
        >>> state_dict = load_state_dict_from_google_drive(id='18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG')
        
        """
    
    if model_dir is None:
        torch_home = torch.hub._get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

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
        filename  = _get_name(id) # use default name of the file in gdrive

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):   
        url = 'https://drive.google.com/uc?id={}'.format(id)
        gdown.download(url, cached_file, quiet=not(progress))
        
    
    if zipfile.is_zipfile(cached_file):
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            members = cached_zipfile.infolist()
            
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            cached_file = os.path.join(model_dir, extraced_name)
    
    print(cached_file)
    return torch.load(cached_file, map_location=map_location)
 

# READ_DATA_CHUNK = 32768
# HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

# def _download_file_from_google_drive(id, destination):
    
#     response = _get_response(id)
#     filename = _get_name_file(response)
    
#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = _get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)
#         filename = _get_name_file(response)
#         destination = os.path.join(destination, filename)

#     _save_response_content(response, destination)    


# def _get_response(id):
#     URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = _get_confirm_token(response)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)
    
#     return response
    
    
# def _get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value

#     return None


# def _save_response_content(response, destination, hash_prefix=None, progress=True):
    
#     if hash_prefix is not None:
#         sha256 = hashlib.sha256()

#     with open(destination, "wb") as f:
#         for chunk in tqdm(response.iter_content(READ_DATA_CHUNK)):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)
#                 if hash_prefix is not None:
#                     sha256.update(chunk)

#     if hash_prefix is not None:
#         digest = sha256.hexdigest()
#         if digest[:len(hash_prefix)] != hash_prefix:
#             raise RuntimeError('invalid hash value (expected "{}", got "{}")'
#                                .format(hash_prefix, digest))
                        
                      
                
# def _get_name_file(response):
#     return response.headers.get('Content-Disposition').split(';')[1].split('\"')[1]
    

# def load_state_dict_from_google_drive(id, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
#     r"""Loads the Torch serialized object at the given google drive file id.
#     Note: Inspired by torch.hub.load_state_dict_from_url

#     If downloaded file is a zip file, it will be automatically
#     decompressed.

#     If the object is already present in `model_dir`, it's deserialized and
#     returned.
#     The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
#     environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
#     ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
#     filesytem layout, with a default value ``~/.cache`` if not set.

#     Args:
#         url (string): URL of the object to download
#         model_dir (string, optional): directory in which to save the object
#         map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
#         progress (bool, optional): whether or not to display a progress bar to stderr.
#             Default: True
#         check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
#             ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
#             digits of the SHA256 hash of the contents of the file. The hash is used to
#             ensure unique names and to verify the contents of the file.
#             Default: False

#     Example:
#         >>> state_dict = load_state_dict_from_google_drive(id='18KRngGJMAhQJmlzjHmgyXuNjqd2l6rQG')
        
#         """
    
#     if model_dir is None:
#         torch_home = torch.hub._get_torch_home()
#         model_dir = os.path.join(torch_home, 'checkpoints')

#     try:
#         os.makedirs(model_dir)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             # Directory already exists, ignore.
#             pass
#         else:
#             # Unexpected OSError, re-raise.
#             raise
            
    
#     response = _get_response(id)
#     try:
#         filename = _get_name_file(response)
#     except:
#         filename = file_name
    
#     cached_file = os.path.join(model_dir, filename)
#     if not os.path.exists(cached_file):
#         url = 'https://drive.google.com/file/d/{}/view?usp=sharing'.format(id)
#         sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
#         hash_prefix = HASH_REGEX.search(filename).group(1) if check_hash else None
#         _save_response_content(response, cached_file, hash_prefix, progress=progress)
    
    
#     if zipfile.is_zipfile(cached_file):
#         with zipfile.ZipFile(cached_file) as cached_zipfile:
#             members = cached_zipfile.infolist()
            
#             if len(members) != 1:
#                 raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            
#             cached_zipfile.extractall(model_dir)
#             extraced_name = members[0].filename
#             cached_file = os.path.join(model_dir, extraced_name)
    
#     print(cached_file)
#     return torch.load(cached_file, map_location=map_location)
 
