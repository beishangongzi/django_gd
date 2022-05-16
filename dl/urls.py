from django.urls import path
from rest_framework.routers import SimpleRouter, DefaultRouter

from . import views

urlpatterns = [
]

router = DefaultRouter()
router.register("train", viewset=views.TrainView, basename="backbone")

urlpatterns += router.urls
