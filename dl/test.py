# import logging
# import os
# import time
# import json
#
# import torch
# import torchvision
# import numpy as np
# from PIL import Image
#
# from dl.seg.utils import get_device
#
#
# def main(number_class=5,
#          weight_pth='',
#          img="img",
#          ):
#     assert os.path.exists(weight_pth), f"weight path {weight_pth} not found"
#     assert os.path.exists(img), f"img path {img} not found"
#
#     with open("seg/utils/palette.json", "rb") as f:
#         palette_dict = json.load(f)
#         palette = []
#         for v in palette_dict.values():
#             palette += v
#     device = get_device()
#     logging.info(f"using {device}")
#     model =
#
#
