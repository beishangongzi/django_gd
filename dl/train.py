# create by andy at 2022/5/16
# reference:
import datetime
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from dl.seg import utils, config
from dl.seg.model.FcnResnet50 import fcn_resnet50
from dl.seg.dataset.obt_dataset import ObtTrainDataset
from dl.seg.utils.train_and_eval import create_lr_scheduler, train_one_epoch, evaluate

logging.basicConfig(format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def create_model(input_channels, num_classes):
    model = fcn_resnet50(input_channels=input_channels, num_classes=num_classes)

    return model


def train():
    device = utils.get_device()
    num_classes = 5
    input_channels = 32
    batch_size = 8
    lr = 0.001
    weight_decay = 1e-4

    logging.info(f"using {device}")
    logging.info(f"batch_size {batch_size}")

    log_file = os.path.join(config.LOG_DIR, "result.log")
    train_dataset = ObtTrainDataset("image", "train")
    val_dataset = ObtTrainDataset("valImage", "val")
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    model = create_model(input_channels=input_channels, num_classes=num_classes)
    model.to(device)
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 100, warmup=True)
    start_time = time.time()
    for epoch in range(100):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=10, scaler=None)
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
    torch.save(config.LOG_DIR, "save_weights/model_{}.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
if __name__ == '__main__':
    train()
