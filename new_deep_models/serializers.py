# create by andy at 2022/5/14
# reference:

from rest_framework import serializers

from . import models

# class TrainSerializer(serializers.Serializer):
#     model = serializers.CharField(max_length=10, help_text="model name")
#     train_data = serializers.CharField(max_length=100, help_text="train data")
#     val_data = serializers.CharField(max_length=100, help_text="validate data")
#     batch = serializers.IntegerField(default=32, help_text="batch size")
#     epoch = serializers.IntegerField(default=100, help_text="epoch default 100")
#     lr = serializers.FloatField(default=0.001, help_text="learning rate")
#     decay_rate = serializers.FloatField(default=0)
#     save_prefix = serializers.CharField(default="a", help_text="prefix of save name")
#     flush_secs = serializers.IntegerField(default=30, help_text="tensorboard flush")

class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Train
        fields = "__all__"

if __name__ == '__main__':
    pass
