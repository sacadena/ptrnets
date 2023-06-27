from unittest.mock import patch

import pytest

from ptrnets import zoo
from ptrnets.utils.config import get_config_file
from ptrnets.utils.config import load_state_dict_from_model_name
from ptrnets.utils.config import TomlConfig


@pytest.fixture
def config_file_path():
    return "config.toml"


@pytest.fixture
def mock_toml_config(config_file_path):
    with patch("ptrnets.utils.config.TomlConfig") as mock_config:
        mock_config.return_value._file_path = config_file_path
        yield mock_config


@pytest.fixture
def mock_load_url():
    with patch("ptrnets.utils.config.load_state_dict_from_url") as mock_load_url:
        yield mock_load_url


def test_init_nonexistent_file(config_file_path):
    with pytest.raises(ValueError):
        TomlConfig("nonexistent.toml")


def test_load_state_dict_from_model_name_url(mock_toml_config, mock_load_url):
    model_name = "model1"
    url = "http://example.com/model1.pth"
    config = mock_toml_config.return_value
    config.get_dict.return_value = {"url": {"model1": url}}
    load_state_dict_from_model_name(model_name)
    mock_toml_config.assert_called_once_with(file_path=get_config_file(package=zoo))
    mock_load_url.assert_called_once_with(url, progress=True, map_location="cpu", check_hash=False)
