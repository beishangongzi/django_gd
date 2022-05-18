from django.urls import path

from .views import schema_view

urlpatterns = [
    path("docs", schema_view.with_ui("swagger", cache_timeout=0), name="schema-swagger",)
]