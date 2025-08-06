from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_view, name='chat'),
    path('get-response/', views.validate_url_view, name='get_message'),
    # New chat history endpoints
    path('chat-history/', views.get_chat_history_view, name='get_chat_history'),
    path('search-history/', views.search_chat_history_view, name='search_chat_history'),
    path('clear-history/', views.clear_chat_history_view, name='clear_chat_history'),
    
]
