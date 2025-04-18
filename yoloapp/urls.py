from django.urls import path
from . import views
from .views import upload_and_predict, user_login, user_logout
from .views import register
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('upload/', views.upload_and_predict, name='upload_and_predict'),
    path('', user_login, name='login'),
    # path('logout/', user_logout, name='logout'),
    path('register/', register, name='register'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]
