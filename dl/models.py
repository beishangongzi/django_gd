import django.utils.timezone
import os
import uuid

from django.contrib.auth.models import User
from django.db import models



class Train(models.Model):
    val_dataset = models.CharField(max_length=100, blank=False, help_text="val dataset")
    train_dataset = models.CharField(max_length=100, blank=False, help_text="train dataset")
    model_name = models.CharField(max_length=100, blank=False, help_text="model name")
    num_classes = models.IntegerField(default=5, help_text="number classes")
    epoch = models.IntegerField(default=300, help_text="number of total epoch")
    input_channels = models.IntegerField(default=32, help_text="number of input's channels")
    batch_size = models.IntegerField(default=32, help_text="number of batch")
    lr = models.FloatField(default=0.001, help_text="learning rate")
    weight_decay = models.FloatField(default=1e-4, help_text="weight decay")
    print_freq = models.IntegerField(default=10, help_text="print freq in console")
    pre_train = models.CharField(default="", max_length=100, help_text=" if pre_train")
    start_epoch = models.IntegerField(default=0, help_text="start epoch default 0")
    is_close = models.BooleanField(default=False, help_text="morphology close, default false")
    is_erode = models.BooleanField(default=False)
    is_dilate = models.BooleanField(default=False)
    is_open = models.BooleanField(default=False)


def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    print(instance.user)
    filename = '{}-{}.{}'.format(filename, uuid.uuid4().hex[:10], ext)
    return os.path.join("files", filename)


class Test(models.Model):
    user =models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    saved_model = models.CharField(max_length=100)
    image = models.FileField(upload_to=user_directory_path, null=False)
    time = models.DateTimeField(default=django.utils.timezone.now)
