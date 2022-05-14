# create by andy at 2022/5/14
# reference:
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
import os

from torch.utils.tensorboard import SummaryWriter

from new_deep_models.deep_learning import config, utils
from new_deep_models.deep_learning.dataset.obt_dataset import ObtTrainDataset
from new_deep_models.deep_learning.model.FcnResnet50 import fcn_resnet50, fcn_resnet101, fcn_resnet152
from new_deep_models.deep_learning.utils.cal_accuracy import SegmentationMetric
from new_deep_models.deep_learning.utils.confusion_matrix import add_confusion_matrix


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train(model,
          train_dataset,
          val_dataset,
          batch_size,
          number_classes,
          epoch,
          lr,
          decay_rate,
          save_name,
          ):
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay_rate)
    dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    old_mIoU = 0
    old_name = None
    for i in range(epoch):
        print("------------{} begin--------------".format(i))
        model.train()
        running_loss = 0.0
        j = 0
        for data in dl:
            j += 1
            inputs, target = data
            inputs = inputs.to(device)
            target = target.to(device)
            target = torch.squeeze(target).long().to(device)

            optimizer.zero_grad()

            try:
                outputs = model(inputs)["out"]
            except:
                outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item()
        print(running_loss / len(dl))

        model.eval()
        with torch.no_grad():
            dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        val_value = valiate(model, dl, number_classes=number_classes)

        val_value.update({"train_loss": running_loss, "epoch": i})
        yield val_value
        if val_value["mIoU"] > old_mIoU:
            old_mIoU = val_value["mIoU"]
            if old_name is not None:
                os.remove(old_name)
            name = os.path.join(os.path.dirname(__file__), f"models/{save_name}_{str(i)}.pth")
            old_name = name
            torch.save(model.state_dict(), name)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"models/{save_name}_last_.pth"))


def valiate(model, val_loader, number_classes):
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    loss_all = 0
    PA = []
    MPA = []
    HIST = torch.zeros([4, 4])
    MIOU = []

    for data in val_loader:
        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)
        label = torch.squeeze(label, 1).long().to(device)

        try:
            outputs = model(inputs)["out"]
        except:
            outputs = model(inputs)

        loss = criterion(outputs, label)
        loss_val = loss.cpu().item()
        loss_all += loss_val

        outputs = torch.argmax(outputs, 1).long()
        label = torch.squeeze(label, 1).long()

        ignore_labels = [number_classes - 1]
        metric = SegmentationMetric(numClass=number_classes)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
        hist = metric.addBatch(outputs.cpu(), label.cpu(), ignore_labels)
        pa = metric.pixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()

        PA.append(pa)
        MPA.append(mpa)
        MIOU.append(mIoU)
        HIST += hist

    loss_all = loss_all / len(val_loader)
    pa_all = np.mean(PA)
    mpa_all = np.mean(MPA)
    mIoU_all = np.mean(MIOU)
    res = {
        "loss": loss_all,

        "pa": pa_all,
        "mpa": mpa_all,
        "mIoU": mIoU_all,
        "hist": HIST,

    }

    return res


def process(output, label):
    morphology_close = utils.Morphology.close(output.cpu().numpy().astype("uint8"))
    morphology_close_color = utils.Utils.to_color(morphology_close)


def new_run(train_dataset_path, val_dataset_path, model, batch_size, epoch, lr, decay_rate, save_prefix, flush_secs, num_classes, input_channels):
    save_name = "-".join([str(save_prefix), model, str(batch_size), str(epoch), str(lr), str(decay_rate)]).replace("/",
                                                                                                                   ".")
    train_dataset_path = os.path.join(config.DATA_DIR, 'obt', train_dataset_path)
    val_dataset_path = os.path.join(config.DATA_DIR, 'obt', val_dataset_path)

    models = {"fcn_resnet50": fcn_resnet50, "fcn_resnet101": fcn_resnet101, "fcn_resnet152": fcn_resnet152}
    model = models[model](num_classes=num_classes, input_channels=input_channels).to(get_device())

    train_dataset = ObtTrainDataset(train_dataset_path)
    val_dataset = ObtTrainDataset(val_dataset_path)
    history = train(model, train_dataset, val_dataset, batch_size, num_classes, epoch, lr, decay_rate, save_name)

    writer = SummaryWriter(os.path.join(config.LOG_DIR, save_name), comment=save_name, flush_secs=flush_secs,
                           filename_suffix="aa")
    for record in history:
        loss_train = record["train_loss"]
        loss_val = record["loss"]
        pa = record['pa']
        mpa = record['mpa']
        mIoU = record["mIoU"]
        hist = record["hist"]
        epoch = record["epoch"]
        if epoch < 10:
            continue
        writer.add_scalars("loss", {"loss_train": loss_train, "loss_val": loss_val}, epoch)
        writer.add_scalar("pa", pa, epoch)
        writer.add_scalar("map", mpa, epoch)
        writer.add_scalar("mIoU", mIoU, epoch)
        add_confusion_matrix(writer, hist, 4, class_names=['a', "b", 'c', 'd'], global_step=epoch)

    writer.close()


if __name__ == '__main__':
    # run("image", "image", "fcn_resnet50", 8, 10, 0.001, 0.001, "a", 2)
    pass
