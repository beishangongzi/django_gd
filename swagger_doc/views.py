from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions


schema_view = get_schema_view(
    openapi.Info(
        title="接口文档",
        default_version="1.0",
        terms_of_service='',
        contact=openapi.Contact(name="Andy Z Wright", email="andyzwright021@gmail.com"),
        license=openapi.License(name="MIT LICENCE"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,)
)