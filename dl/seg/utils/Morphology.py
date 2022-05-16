# create by andy at 2022/5/9
# reference:
import numpy as np

import cv2
import torch


class Morphology:
    @staticmethod
    def fun(img, kernel_size, iterations, operation):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = img.astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        output = cv2.morphologyEx(img, operation, kernel, iterations=iterations)
        return torch.tensor(output).cpu()

    @staticmethod
    def close(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_CLOSE)

    @staticmethod
    def open(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_OPEN)

    @staticmethod
    def dilate(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_DILATE)

    @staticmethod
    def erode(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_ERODE)
