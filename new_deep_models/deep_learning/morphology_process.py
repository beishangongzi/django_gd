import torch.nn as nn
import torch
import numpy as np

from new_deep_models.deep_learning.utils.cal_accuracy import SegmentationMetric
from new_deep_models.deep_learning.utils.get_device import get_device


def valiate(model, val_loader, number_classes, morphology):
    device = get_device()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4037,  0.7134,  1.2551, 40.8860,  3.3320])).to(device)
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

        label = torch.squeeze(label, 1).long()

        ignore_labels = []
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
