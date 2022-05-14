from django.db import models


# Create your models here.

class Train(models.Model):
    model = models.CharField('model', max_length=100, help_text="model name")
    train_dataset_path = models.CharField('train_dataset_path', max_length=100, help_text="train data")
    val_dataset_path = models.CharField("val_dataset_path", max_length=100, help_text="validate data")
    batch_size = models.IntegerField("batch_size", default=32, help_text="batch size")
    epoch = models.IntegerField("epoch", default=100, help_text="epoch default 100")
    lr = models.FloatField("lr", default=0.001, help_text="learning rate")
    decay_rate = models.FloatField("decay_rate", default=0)
    save_prefix = models.CharField("save_prefix", default="a", max_length=10, help_text="prefix of save name")
    flush_secs = models.IntegerField("flush_secs", default=30, help_text="tensorboard flush")
