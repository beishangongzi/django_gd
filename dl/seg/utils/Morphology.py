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


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np


    def image_RGB2Gray(image_path = "./fusion_datasets/1.jpg"):
        # 图片路径，相对路径

        # 读取图片
        image = Image.open(image_path)
        # 输出维度
        print("RGB图像的维度：", np.array(image).shape)
        # 显示原图
        image.show()
        # RGB转换我灰度图像
        image_transforms = transforms.Compose([
            transforms.Grayscale(1)
        ])
        image = image_transforms(image)
        # 输出灰度图像的维度
        print("灰度图像维度： ", np.array(image).shape)
        # 显示灰度图像
        image.show()
        return image


    test_image = "img.png"
    img = image_RGB2Gray(test_image)
    img.show()
    img = np.array(img)
    img_close = Morphology.close(img, 5, 2)
    img_close = Image.fromarray(img_close.cpu().numpy())
    img_close.show("close")

    img_close = Morphology.erode(img, 5, 2)
    img_close = Image.fromarray(img_close.cpu().numpy())
    img_close.show("erode")

    img_close = Morphology.dilate(img, 5, 2)
    img_close = Image.fromarray(img_close.cpu().numpy())
    img_close.show("dilate")

    img_close = Morphology.open(img, 5, 2)
    img_close = Image.fromarray(img_close.cpu().numpy())
    img_close.show("open")

