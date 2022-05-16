# create by andy at 2022/5/13
# reference:
from typing import Any

from torch.nn import Conv2d
from torchvision.models import mobilenetv3
from torchvision.models.segmentation.lraspp import LRASPP, _lraspp_mobilenetv3


def lraspp_mobilenet_v3_large(
        num_classes: int = 21,
        in_channels=32,
        **kwargs: Any,
) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """

    backbone = mobilenetv3.mobilenet_v3_large(pretrained=False, dilated=True)
    backbone.features[0][0] = Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model = _lraspp_mobilenetv3(backbone, num_classes)

    return model


if __name__ == '__main__':
    # writer = SummaryWriter(config.LOG_DIR)
    a = lraspp_mobilenet_v3_large(5, 32)
    print("")
    # writer.add_graph(a, test_data, verbose=False, use_strict_trace=False)
    # writer.close()

    # a = deeplabv3_resnet50()
    # test_data = torch.zeros([8, 3, 128, 128])
    # res = a(test_data)["out"]
    # print(res)
