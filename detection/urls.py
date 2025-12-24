from django.urls import path
from . import views  # Import file views.py của bạn

urlpatterns = [
    # URL gốc (giữ nguyên)
    path('', views.home, name='home'),
    path('video_feed', views.video_feed, name='video_feed'),

    # ⭐ 2 URL MỚI CHO API
    path('api/get-last-detection/', views.api_get_last_detection, name='api_get_last_detection'),
    path('api/reset-detection/', views.api_reset_detection, name='api_reset_detection'),
]