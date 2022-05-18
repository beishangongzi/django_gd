from rest_framework import status
from rest_framework.viewsets import GenericViewSet as View, ModelViewSet
from rest_framework.response import Response
from rest_framework import mixins

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




class TestView(View, mixins.ListModelMixin, mixins.CreateModelMixin):
    queryset = models.Test.objects.all()
    serializer_class = serializers.TestSerializer

    def list(self, request, *args, **kwargs):
        print(request)
        print(args)
        print(kwargs)
        return super(TestView, self).list(request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        return super(TestView, self).create(request, *args, **kwargs)
