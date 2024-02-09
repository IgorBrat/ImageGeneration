from django.db import models


class Images(models.Model):
    content_image = models.ImageField()
    style_image = models.ImageField()

    class Meta:
        app_label = 'web'

    def __str__(self):
        return 'images'
