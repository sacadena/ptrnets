import requests

from ptrnets.utils.gdrive import _get_name


def test_get_name_with_requests_session(mocker):
    mock_get = mocker.patch.object(requests.Session, "get")
    mock_get.return_value.headers = {"Content-Disposition": 'attachment; filename="example_model.pth"'}

    name = _get_name("example_id")

    mock_get.assert_called_once_with("https://drive.google.com/uc?id=example_id", stream=True)
    assert name == "example_model.pth"
