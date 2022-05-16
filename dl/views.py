from rest_framework import status
from rest_framework.viewsets import GenericViewSet as View
from rest_framework.response import Response

from django_gd.celery import run_new_model, run_dl_model
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
        run_dl_model.delay(**self.serializer_class(instance).data)
        return Response(self.serializer_class(instance).data)


    def retrieve(self, request, pk):
        instance = self.get_object()
        return Response(self.get_serializer(instance).data)

    def update(self, request, pk):
        instance = self.get_object()
        serializer = self.get_serializer(instance, request.data)
        serializer.is_valid(True)
        return Response(self.get_serializer(serializer.save()).data)

    def partial_update(self, request, pk):
        instance = self.get_object()
        serializer = self.get_serializer(instance, request.data, partial=True)
        serializer.is_valid(True)
        return Response(self.get_serializer(serializer.save()).data)

    def destroy(self, request, pk):
        instance = self.get_object()
        instance.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)