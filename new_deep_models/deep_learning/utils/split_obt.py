import os
import shutil

import numpy as np

from new_deep_models.deep_learning import config

data_dir = os.path.join(config.DATA_DIR, "obt_raw")
obt_dir = os.path.join(config.DATA_DIR, config.DATAS["obt"])
try:
    shutil.rmtree(obt_dir)
except FileNotFoundError:
    pass
items = os.listdir(data_dir)
os.makedirs(os.path.join(obt_dir, "image"))
os.makedirs(os.path.join(obt_dir, "imageMasks"))
x_path = os.path.join(obt_dir, "image")
y_path = os.path.join(obt_dir, "imageMasks")
print(items)

for item in items:
    print(item)
    Y = np.load(os.path.join(data_dir, item), allow_pickle=True).item().get("segmentation_mask")
    X = np.load(os.path.join(data_dir, item), allow_pickle=True).item().get("image")
    Y = Y - 1
    edge = [0, 128, 256, -256, -128, None]
    for i in [0, 1, 3, 4]:
        for j in [0, 1, 3, 4]:
            y1 = Y[edge[i]:edge[i+1], edge[j]:edge[j+1], 0]
            x1 = X[edge[i]:edge[i+1], edge[j]:edge[j+1], :]
            name = item.split(".")[0] + "_{}_{}".format(i, j)
            np.save(os.path.join(x_path, name), x1)
            np.save(os.path.join(y_path, name), y1)

if __name__ == '__main__':
    pass