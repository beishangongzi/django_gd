import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader

from new_deep_models.deep_learning.utils import Morphology
from new_deep_models.deep_learning.utils.cal_accuracy import SegmentationMetric
from new_deep_models.deep_learning.utils.get_device import get_device


def cal_metrics(output, label, ignore_labels, number_classes):
    metric = SegmentationMetric(numClass=number_classes)  # 3表示有3个分类，有几个分类就填几, 0也是1个分类
    hist = metric.addBatch(output.cpu(), label.cpu(), ignore_labels)
    pa = metric.pixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    return [hist, pa, mpa, mIoU]


def valiate(model, val_dataset, number_classes, morphology_way):
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, drop_last=True)
    device = get_device()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4037, 0.7134, 1.2551, 40.8860, 3.3320])).to(device)
    loss_all = 0
    PA = []
    MPA = []
    HIST = torch.zeros([number_classes, number_classes])
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
        if morphology_way == "close":
            outputs = Morphology.close(outputs, 2, 1)
        elif morphology_way == "open":
            outputs = Morphology.open(outputs, 2, 1)
        elif morphology_way == "erode":
            outputs = Morphology.erode(outputs, 2, 1)
        elif morphology_way == "dilate":
            outputs = Morphology.dilate(outputs, 2, 1)
        else:
            pass
        label = torch.squeeze(label, 1).long()

        ignore_labels = []
        hist, pa, mpa, mIoU = cal_metrics(outputs.cpu(), label.cpu(), ignore_labels, number_classes)

        PA.append(pa)
        MPA.append(mpa)
        MIOU.append(mIoU)
        HIST += hist

    loss_all = loss_all / len(val_loader)
    pa_all = np.mean(PA)
    mpa_all = np.mean(MPA)
    mIoU_all = np.mean(MIOU)
    res = {
        "loss_"+morphology_way: loss_all,

        f"pa_{morphology_way}": pa_all,
        f"mpa_{morphology_way}": mpa_all,
        f"mIoU_{morphology_way}": mIoU_all,
        f"hist_{morphology_way}": HIST,
        f"image_{morphology_way}": outputs,
        f"image_label_{morphology_way}": label

    }

    return res
