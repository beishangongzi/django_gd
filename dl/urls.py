from django.urls import path
from rest_framework.routers import SimpleRouter, DefaultRouter

from . import views

urlpatterns = [
]

router = DefaultRouter()
router.register("train", viewset=views.TrainView, basename="backbone")
router.register("test", viewset=views.TestView, basename="test")

urlpatterns += router.urls
