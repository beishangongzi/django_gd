# create by andy at 2022/5/13
# reference:
from typing import Optional

from torch.nn import Conv2d
from torchvision.models import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_resnet


def deeplabv3(
        num_classes: int = 5,
        input_channels: int = 32,
        backbone: Optional[str] = "resnet50",
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> DeepLabV3:
    if backbone == "resnet50":
        backbone = resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    elif backbone == "resnet152":
        backbone = resnet.resnet152(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    elif backbone == "resnet101":
        backbone = resnet.resnet152(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    else:
        raise ValueError("must in resnet50, resnet101, resnet152")
    backbone.conv1 = Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    return model


def deeplabv3_resnet50(
        num_classes: int = 5,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> DeepLabV3:
    return deeplabv3(num_classes, input_channels, "resnet50", aux_loss, pretrained_backbone, **kwargs)


def deeplabv3_resnet152(
        num_classes: int = 4,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> DeepLabV3:
    return deeplabv3(num_classes, input_channels, "resnet152", aux_loss, pretrained_backbone, **kwargs)


def deeplabv3_resnet101(
        num_classes: int = 4,
        input_channels: int = 32,
        aux_loss: Optional[bool] = None,
        pretrained_backbone: bool = False,
        **kwargs
) -> DeepLabV3:
    return deeplabv3(num_classes, input_channels, "resnet101", aux_loss, pretrained_backbone, **kwargs)


if __name__ == '__main__':
    import torch

    # writer = SummaryWriter(config.LOG_DIR)
    a = deeplabv3_resnet101(5, 32)
    test_data = torch.zeros([8, 32, 128, 128])
    res = a(test_data)["out"]
    print(res)

    # writer.add_graph(a, test_data, verbose=False, use_strict_trace=False)
    # writer.close()

    # a = deeplabv3_resnet50()
    # test_data = torch.zeros([8, 3, 128, 128])
    # res = a(test_data)["out"]
    # print(res)
