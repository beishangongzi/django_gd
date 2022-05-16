# create by andy at 2022/5/14
# reference:
import enum

from . import FcnResnet50, LR_ASP, Deeplabv3


class DPModels(enum.Enum):
    fcn_resnet152 = FcnResnet50.fcn_resnet152
    fcn_resnet101 = FcnResnet50.fcn_resnet101
    fcn_resnet50 = FcnResnet50.fcn_resnet50
    lraspp_mobilenet_v3_large = LR_ASP.lraspp_mobilenet_v3_large
    deeplabv3_resnet152 = Deeplabv3.deeplabv3_resnet152
    deeplabv3_resnet101 = Deeplabv3.deeplabv3_resnet101
    deeplabv3_resnet50 = Deeplabv3.deeplabv3_resnet50



if __name__ == '__main__':
    pass
