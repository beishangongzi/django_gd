from rest_framework.routers import SimpleRouter, DefaultRouter

from . import views

urlpatterns = [
]

router = DefaultRouter()
router.register("users", viewset=views.UserAddView, basename="users")
router.register("login", viewset=views.LoginView, basename="login")
router.register("login-phone", viewset=views.LoginPhoneView, basename="login-phone")
router.register("logout", viewset=views.LoginOutView, basename="logout")

urlpatterns += router.urls
