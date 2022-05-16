# create by andy at 2022/5/16
# reference:
import datetime
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dl.seg import utils, config
from dl.seg.model import DPModels
from dl.seg.dataset.obt_dataset import ObtTrainDataset
from dl.seg.utils.train_and_eval import create_lr_scheduler, train_one_epoch, evaluate

logging.basicConfig(format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def create_model(input_channels, num_classes, name):
    model = getattr(DPModels, name)(input_channels=input_channels, num_classes=num_classes)

    return model


def my_eval(i, model, val_loader, device, num_classes, writer, morphology):
    print("******************open****************************")
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, morphology_way=morphology)
    acc_global, acc, iu = confmat.compute()
    writer.add_scalar("acc_global" + morphology, acc_global, i)
    writer.add_scalar("acc" + morphology, acc, i)
    writer.add_scalar("iu" + morphology, iu, i)
    writer.add_scalar("mean_iu" + morphology, iu.mean().item() * 100, i)

    val_info = str(confmat)
    print(val_info)


def train(
        val_dataset="valImage",
        train_dataset="image",
        model_name="fcn_resnet50",
        num_classes=5,
        epoch=100,
        input_channels=32,
        batch_size=8,
        lr=0.001,
        weight_decay=1e-4,
        print_freq=10):
    device = utils.get_device()
    saved_model_name = "-".join([model_name,
                                 batch_size.__str__(),
                                 lr.__str__(),
                                 weight_decay.__str__(),
                                 ])
    logging.info(f"using {device}")
    logging.info(f"batch_size {batch_size}")

    log_file = os.path.join(config.LOG_DIR, "result.log")
    train_dataset = ObtTrainDataset(train_dataset, "train")
    val_dataset = ObtTrainDataset(val_dataset, "val")
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    model = create_model(input_channels=input_channels, num_classes=num_classes, name=model_name)
    model.to(device)
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epoch, warmup=True)
    start_time = time.time()
    writer = SummaryWriter(os.path.join(config.LOG_DIR, saved_model_name), comment=saved_model_name, flush_secs=10)
    for i in range(epoch):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, i,
                                        lr_scheduler=lr_scheduler, print_freq=print_freq, scaler=None)
        print(mean_loss)

        my_eval(i, model, val_loader, device, num_classes, writer, "")
        my_eval(i, model, val_loader, device, num_classes, writer, "open")
        my_eval(i, model, val_loader, device, num_classes, writer, "close")
        my_eval(i, model, val_loader, device, num_classes, writer, "erode")
        my_eval(i, model, val_loader, device, num_classes, writer, "dilate")

    writer.close()

    torch.save(config.LOG_DIR, f"model_{saved_model_name}.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    train()
