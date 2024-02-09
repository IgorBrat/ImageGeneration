from django.contrib import admin
from django.urls import path
from .views import general, train, inference
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # GENERAL PAGES
    path('', general.home, name="home"),
    path('train/', general.train, name="train"),
    path('infer/', general.inference, name="infer"),
    path('transfer/', general.style_transfer, name="style-transfer"),
    path('logs/', general.logs, name="logs"),

    # LOGGING
    path('logging/<str:log_file>', general.logging, name="logging"),
    path('logs/<str:log_file>', general.log, name="log"),

    # MODELS TRAINING
    path('train/<str:model_name>/', train.train, name="train-model"),

    # INFERENCE
    path('infer/<str:model_name>/<str:version>/', inference.infer, name="infer-model"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
