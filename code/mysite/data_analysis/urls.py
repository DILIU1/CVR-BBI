from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_signal, name='upload_signal'),
    path('submit/', views.submit, name='submit_data'),
    path('analysis/', views.analyze_data, name='analyze_data'),
    path('analysisR/', views.upload_data_matrix, name='upload_data_matrix'),
]
