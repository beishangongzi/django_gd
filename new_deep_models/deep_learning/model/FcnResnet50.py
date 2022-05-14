# create by andy at 2022/5/13
# reference:
from typing import Optional

import torch
from torch.nn import Conv2d
from torchvision.models import resnet
from torchvision.models.segmentation.fcn import _fcn_resnet, FCN


def fcn(
        num_classes: int = 5,
        input_channels: int = 32,
        backbone: Optional[str] = "resnet50",
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> FCN:
    if backbone == "resnet50":
        backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    elif backbone == "resnet152":
        backbone = resnet.resnet152(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    elif backbone == "resnet101":
        backbone = resnet.resnet152(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    else:
        raise ValueError("must in resnet50, resnet101, resnet152")
    backbone.conv1 = Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    return model


def fcn_resnet50(
        num_classes: int = 5,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> FCN:
    return fcn(num_classes, input_channels, "resnet50", aux_loss, pretrained_backbone, **kwargs)


def fcn_resnet152(
        num_classes: int = 4,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> FCN:
    return fcn(num_classes, input_channels, "resnet152", aux_loss, pretrained_backbone, **kwargs)


def fcn_resnet101(
        num_classes: int = 4,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> FCN:
    return fcn(num_classes, input_channels, "resnet101", aux_loss, pretrained_backbone, **kwargs)


if __name__ == '__main__':
    a = fcn_resnet101(5, 32)
    test_data = torch.zeros([1, 32, 128, 128])
    output = a(test_data)["out"]
    print(output.size())
