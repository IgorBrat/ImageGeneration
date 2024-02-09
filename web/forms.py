from django import forms
from .models import Images


class ImageForm(forms.ModelForm):
    """Form for the image model"""

    class Meta:
        model = Images
        fields = ('content_image', 'style_image')
