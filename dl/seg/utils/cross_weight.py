import os

import numpy as np
import torch
from sklearn.utils import class_weight

from dl.seg import config

data_dir = os.path.join(config.DATA_DIR, "obt_raw")
y_all = []
for item in os.listdir(data_dir):
    y = np.load(os.path.join(data_dir, item), allow_pickle=True).item().get("segmentation_mask")
    y = y.flatten()
    y = y.tolist()
    y_all.append(y)
y_all = np.array(y_all).flatten()
print(np.unique(y_all, return_counts=True))
# class_weights=class_weight.compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
# class_weights=torch.tensor(class_weights, dtype=torch.float)
# print(np.unique(y_all))
# print(class_weights)