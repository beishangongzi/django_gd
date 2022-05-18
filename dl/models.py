import django.utils.timezone
import os
import uuid

from django.contrib.auth.models import User
from django.db import models



class Train(models.Model):
    val_dataset = models.CharField(max_length=100, blank=False)
    train_dataset = models.CharField(max_length=100, blank=False)
    model_name = models.CharField(max_length=100, blank=False)
    num_classes = models.IntegerField(default=5)
    epoch = models.IntegerField(default=300)
    input_channels = models.IntegerField(default=32)
    batch_size = models.IntegerField(default=32)
    lr = models.FloatField(default=0.001)
    weight_decay = models.FloatField(default=1e-4)
    print_freq = models.IntegerField(default=10)
    pre_train = models.CharField(default="", max_length=100)
    start_epoch = models.IntegerField(default=0)
    is_close = models.BooleanField(default=False)
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
