from django.urls import path

from .views import schema_view
from rest_framework.documentation import include_docs_urls
from rest_framework.permissions import AllowAny

urlpatterns = [
    path("docs", schema_view.with_ui("swagger", cache_timeout=0), name="schema-swagger",),
    path("doc", include_docs_urls(title="api rest frame work", permission_classes=[AllowAny]))
]