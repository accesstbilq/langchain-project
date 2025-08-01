from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_view, name='chat'),
    path('get-response/', views.validate_url_view, name='get_message'),
]
