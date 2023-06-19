import torch
from torchvision.models import alexnet
from torchvision.models import resnet50
from torchvision.models import vgg16

from ..utils.config import load_state_dict_from_model_name

__all__ = [
    "resnet50_trained_on_sin",
    "resnet50_trained_on_sin_and_in",
    "vgg16_trained_on_sin",
    "resnet50_trained_on_sin_and_in_then_finetuned_on_in",
    "alexnet_trained_on_sin",
]


def _model(model_name, model_fn, pretrained, progress, use_data_parallel, **kwargs):
    model = model_fn(pretrained=False)

    if "vgg" in model_name:
        model.features = (
            torch.nn.DataParallel(model.features)
            if use_data_parallel
            else model.features
        )
    else:
        model = torch.nn.DataParallel(model) if use_data_parallel else model

    if pretrained:
        checkpoint = load_state_dict_from_model_name(model_name, progress=progress)

        if use_data_parallel:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            new_dict = dict()
            for k, v in checkpoint["state_dict"].items():
                new_dict.update({k.replace("module.", ""): v})
            model.load_state_dict(new_dict)

    return model


def resnet50_trained_on_sin(
    pretrained: bool = True,
    progress: bool = True,
    use_data_parallel: bool = False,
    **kwargs,
):
    r"""Resnet50 trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX
    """
    return _model(
        "resnet50_trained_on_sin",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_trained_on_sin_and_in(
    pretrained: bool = True,
    progress: bool = True,
    use_data_parallel: bool = False,
    **kwargs,
):
    r"""Resnet50 trained on Sylized ImageNet + standard ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX
    """
    return _model(
        "resnet50_trained_on_sin_and_in",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def resnet50_trained_on_sin_and_in_then_finetuned_on_in(
    pretrained: bool = True,
    progress: bool = True,
    use_data_parallel: bool = False,
    **kwargs,
):
    r"""Resnet50 trained on Sylized ImageNet + standard ImageNet. Then finetuned on ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX
    """
    return _model(
        "resnet50_trained_on_sin_and_in_then_finetuned_on_in",
        resnet50,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )


def vgg16_trained_on_sin(
    pretrained: bool = True,
    progress: bool = True,
    use_data_parallel: bool = False,
    **kwargs,
):
    r"""Vgg16 trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX
    """
    return _model(
        "vgg16_trained_on_sin", vgg16, pretrained, progress, use_data_parallel, **kwargs
    )


def alexnet_trained_on_sin(
    pretrained: bool = True,
    progress: bool = True,
    use_data_parallel: bool = False,
    **kwargs,
):
    r"""AlexNet trained on Sylized ImageNet as in Geirhos et al 2019:
    "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy
    and robustness" (ICLR 2019 Oral) https://openreview.net/forum?id=Bygh9j09KX
    """
    return _model(
        "alexnet_trained_on_sin",
        alexnet,
        pretrained,
        progress,
        use_data_parallel,
        **kwargs,
    )
