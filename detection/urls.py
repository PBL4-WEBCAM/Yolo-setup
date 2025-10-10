from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('api/detect/', views.detect_api, name='detect_api'),
    path('api/health/', views.health_check, name='health_check'),
]