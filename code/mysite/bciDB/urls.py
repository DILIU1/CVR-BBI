from django.urls import path

from . import views
urlpatterns = [
    path("", views.trial_data_list, name="database"),
]