from rest_framework import status
from rest_framework.viewsets import GenericViewSet as View
from rest_framework.response import Response

from django_gd.celery import run_new_model
from . import serializers
from . import models


class TrainView(View):
    queryset = models.Train.objects.all()
    serializer_class = serializers.TrainSerializer

    def list(self, request):
        serializers = self.serializer_class(self.get_queryset(), many=True)
        return Response(serializers.data)

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(True)
        instance = serializer.save()
        print(self.serializer_class(instance).data)
        run_new_model.delay(**self.serializer_class(instance).data)
        return Response(self.serializer_class(instance).data)
