# encoding: utf-8
import logging
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from dl.seg.Transforms import Squeeze
from dl.seg import config


class ObtTrainDataset(Dataset):
    def __init__(self, image_path=r"image", mode="train"):
        self.name = "obt"
        assert mode in ("train", "val", "test")
        self.image_path = os.path.join(config.DATA_DIR, config.DATAS[self.name], image_path)
        self.image_list = glob(os.path.join(self.image_path, "*.npy"))
        if len(self.image_list) == 0:
            logging.warning(f"the dataset's length is 0. the image path {image_path}")
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path + "Masks"

        self.transform_x = T.Compose(
            [T.ToTensor()])
        self.transform_mask = T.Compose([T.ToTensor(), Squeeze()])

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = os.path.basename(self.image_list[index])

            X = np.load(os.path.join(self.image_path, image_name))
            masks = np.load(os.path.join(self.mask_path, image_name))
            X = X / 1.0
            masks = masks / 1.0
            X = self.transform_x(X)
            masks = self.transform_mask(masks)
            X = X.type(torch.FloatTensor)
            masks = masks.type(torch.LongTensor)
            return X, masks

        else:
            image_name = os.path.basename(self.image_list[index])
            X = np.load(os.path.join(self.image_path, image_name))
            X = X / 1.0
            X = self.transform_x(X)
            X = X.type(torch.FloatTensor)
            return X, image_name

    def __len__(self):
        return len(self.image_list)


def test():
    train_data = ObtTrainDataset()
    print(len(train_data))
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_labels.size())


if __name__ == '__main__':
    test()
