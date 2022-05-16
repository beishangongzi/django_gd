# create by andy at 2022/5/16
from rest_framework import serializers
from . import models


class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Train
        fields = "__all__"
