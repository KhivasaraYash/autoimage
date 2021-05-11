
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home,name='home'),
    path('upload',views.uploadImages,name='uploadImages'),
    path('loadmodel',views.loadModel,name='run_model'),
    path('runmodel',views.runmodel),
    path('showimages',views.showResults),
    path('save',views.download,name='download_res'),
    path('automatic',views.switchAuto),
    path('manual',views.switchMan),
    path('manual_upload',views.manualUpload),
    path('uploadmodelfiles',views.customModel),
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)