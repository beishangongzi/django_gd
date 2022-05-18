from django.contrib.auth.models import User
from django.db import models


# Create your models here.

class UserAdd(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    phone = models.IntegerField()

    def __str__(self):
        return self.user


