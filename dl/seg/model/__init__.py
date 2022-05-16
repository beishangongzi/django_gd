# create by andy at 2022/5/14
# reference:
import enum

from . import FcnResnet50


class DPModels(enum.Enum):
    fcn_resnet152: FcnResnet50.fcn_resnet152
    fcn_resnet101: FcnResnet50.fcn_resnet101
    fcn_resnet50: FcnResnet50.fcn_resnet50


if __name__ == '__main__':
    pass
