from django.db import models


# Create your models here.

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

