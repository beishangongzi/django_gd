# create by andy at 2022/5/14
# reference:
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
from new_deep_models.deep_learning.utils.get_device import get_device
from new_deep_models.deep_learning.val import valiate


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
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4037, 0.7134, 1.2551, 40.8860, 3.3320])).to(device)
    optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=decay_rate)

    dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    old_mIoU = 0
    old_name = None
    for i in range(epoch):
        print("------------{} begin--------------".format(i))
        model.train()
        running_loss = 0.0
        j = 0
        with torch.enable_grad():
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
            val_value = valiate(model, val_dataset, number_classes=number_classes, morphology_way='')
            val_value.update(valiate(model, val_dataset, number_classes=number_classes, morphology_way='close'))
            val_value.update(valiate(model, val_dataset, number_classes=number_classes, morphology_way='erode'))
            val_value.update(valiate(model, val_dataset, number_classes=number_classes, morphology_way='open'))
            val_value.update(valiate(model, val_dataset, number_classes=number_classes, morphology_way='dilate'))
            val_value.update({"train_loss": running_loss, "epoch": i})

            yield val_value

        if val_value["mIoU_"] > old_mIoU:
            old_mIoU = val_value["mIoU_"]
            if old_name is not None:
                os.remove(old_name)
            name = os.path.join(os.path.dirname(__file__), f"models/{save_name}_{str(i)}.pth")
            old_name = name
            torch.save(model.state_dict(), name)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"models/{save_name}_last_.pth"))


def s(dict_, keys):
    res = {}
    for key in keys:
        res.update({key: dict_.get(key)})
    return res


def add_scalar(writer, record, key):
    epoch = record["epoch"]
    loss_dict = s(record, [f"{key}_", f"{key}_close", f"{key}_open", f"{key}_erode", f"{key}_dilate"])

    writer.add_scalars(f"{key}",
                       loss_dict,
                       epoch)

def add_image(record, key, writer):
    epoch = record["epoch"]
    image_dict = s(record, [f"{key}_", f"{key}_close", f"{key}_open", f"{key}_erode", f"{key}_dilate"])
    for item in image_dict:
        writer.add_images(item, utils.Utils.to_colors(image_dict[item]), epoch, dataformats="NHWC")

def new_run(train_dataset_path, val_dataset_path, model, batch_size, epoch, lr, decay_rate, save_prefix, flush_secs,
            num_classes, input_channels):
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
        add_scalar(record=record, key="loss", writer=writer)
        add_scalar(record=record, key="pa", writer=writer)
        add_scalar(record=record, key="mpa", writer=writer)
        add_scalar(record=record, key="mIoU", writer=writer)
        add_confusion_matrix(writer, record["hist_"], num_classes, global_step=record["epoch"], tag="hist")
        add_confusion_matrix(writer, record["hist_erode"], num_classes, global_step=record["epoch"], tag="hist_erode")
        add_confusion_matrix(writer, record["hist_dilate"], num_classes, global_step=record["epoch"], tag="hist_dilate")
        add_confusion_matrix(writer, record["hist_close"], num_classes, global_step=record["epoch"], tag="hist_close")
        add_confusion_matrix(writer, record["hist_open"], num_classes, global_step=record["epoch"], tag="hist_open")
        add_image(record=record, key="image", writer=writer)
        add_image(record=record, key="image_label", writer=writer)

    writer.close()


if __name__ == '__main__':
    new_run(train_dataset_path="image", val_dataset_path='image',
            model="fcn_resnet50", batch_size=32, epoch=100,
            lr=0.001, decay_rate=0.001, save_prefix="a", flush_secs=10,
            num_classes=5, input_channels=32)

    pass
