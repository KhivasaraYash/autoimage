from django.core.files.storage import FileSystemStorage
from django.db import models
import os
from django.conf import settings
# Create your models here.


# function to override save location of image.
def content_file_name(instance, filename):

    upload_dir = 'result/' + instance.imageLabel
    return os.path.join(upload_dir, filename)



# result table to hold result info
class Result(models.Model):

    imageLabel = models.CharField(max_length=100)
    imageFile = models.ImageField(upload_to=content_file_name)

# path to custom model directory i.e model/custom
model_storage = FileSystemStorage(location=settings.MODEL_ROOT)

# storing custom model files here
class CustomModelFiles(models.Model):
    files = models.FileField(storage=model_storage)


