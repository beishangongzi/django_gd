import os
import shutil

import numpy as np

from dl.seg import config

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
    # mask = np.squeeze(Y)
    # mask = (mask != 0)
    # mask = np.stack([mask] * 32, axis=2)
    # X = mask * X
    # edge = [0, 128, 256, -256, -128, None]
    # for i in [0, 1, 3, 4]:
    #     for j in [0, 1, 3, 4]:
    #         y1 = Y[edge[i]:edge[i+1], edge[j]:edge[j+1], 0]
    #         x1 = X[edge[i]:edge[i+1], edge[j]:edge[j+1], :]
    #         name = item.split(".")[0] + "_{}_{}".format(i, j)
    #         np.save(os.path.join(x_path, name), x1)
    #         np.save(os.path.join(y_path, name), y1)


    y1 = Y[0:256, 0:256, 0]
    x1 = X[0:256, 0:256, :]
    name = item.split(".")[0] + "_0"
    np.save(os.path.join(x_path, name), x1)
    np.save(os.path.join(y_path, name), y1)

    y1 = Y[0:256, -256:, 0]
    x1 = X[0:256, -256:, :]
    name = item.split(".")[0] + "_1"
    np.save(os.path.join(x_path, name), x1)
    np.save(os.path.join(y_path, name), y1)

    y1 = Y[-256:, 0:256, 0]
    x1 = X[-256:, 0:256, :]
    name = item.split(".")[0] + "_2"
    np.save(os.path.join(x_path, name), x1)
    np.save(os.path.join(y_path, name), y1)

    y1 = Y[-256:, -256:, 0]
    x1 = X[-256:, -256:, :]
    name = item.split(".")[0] + "_3"
    np.save(os.path.join(x_path, name), x1)
    np.save(os.path.join(y_path, name), y1)
    
    y1 = Y[128:128+256, 128:128+256, 0]
    x1 = X[128:128+256, 128:128+256, :]
    name = item.split(".")[0] + "_4"
    np.save(os.path.join(x_path, name), x1)
    np.save(os.path.join(y_path, name), y1)

if __name__ == '__main__':
    pass