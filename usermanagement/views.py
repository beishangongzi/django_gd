from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from rest_framework import decorators
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.mixins import CreateModelMixin, RetrieveModelMixin, ListModelMixin
from django.contrib.auth import authenticate, login

from . import models
from . import serializers


class UserAddView(GenericViewSet, CreateModelMixin, ListModelMixin):
    queryset = models.UserAdd.objects.all()
    serializer_class = serializers.UserAddSerializer

    def list(self, request, *args, **kwargs):
        print(request)
        print(args)
        print(kwargs)
        return super(UserAddView, self).list(request, *args, **kwargs)

    def create(self, request, *args, **kwargs):
        return super(UserAddView, self).create(request, *args, **kwargs)

    @decorators.action(methods=["POST"], detail=False)
    def login(self, request, *args, **kwargs):
        print(request.data)
        return Response({"ik": 200})


class LoginView(GenericViewSet):
    queryset = User.objects.all()
    serializer_class = serializers.LoginSerializer

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(True)
        print(serializer.validated_data)
        username = request.data["user"]
        pwd = request.data["pwd"]
        user = authenticate(username=username, password=pwd)
        if user:
            login(request, user)
            return Response({"ok": 200})
        return Response({"ok": 400})


class LoginPhoneView(GenericViewSet):
    """
    create:
        login by phone
    """
    queryset = models.UserAdd.objects.all()
    serializer_class = serializers.LoginPhoneSerializer

    def create(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(True)
        username = self.get_queryset().get(phone=serializer.validated_data["phone"]).username
        pwd = request.data["pwd"]
        user = authenticate(username=username, password=pwd)
        if user:
            login(request, user)
            return Response({"ok": 200})
        return Response({"ok": 400})


class LoginOutView(GenericViewSet):
    permission_classes = [IsAuthenticated]

    def list(self, request):
        logout(request)
        return Response({"ok": 200})