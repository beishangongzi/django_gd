from rest_framework import serializers

from . import models


class UserAddSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.UserAdd
        fields = "__all__"
        ref_name = 'usermanagement_useraddserializer'


class LoginSerializer(serializers.Serializer):
    user = serializers.CharField()
    pwd = serializers.CharField()

    class Meta:
        ref_name = 'usermanagement_loginSerializer'


class LoginPhoneSerializer(serializers.Serializer):
    phone = serializers.CharField(help_text="user phone")
    pwd = serializers.CharField(help_text="user password")

    class Meta:
        ref_name = 'usermanagement_loginPhoneSerializer'
