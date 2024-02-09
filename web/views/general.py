import base64

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect
import os

from ml.utils.data_management import transform_unnorm
from ..forms import ImageForm
from PIL import Image
from torchvision import transforms
import torch
from ml.utils.visual import show_images_norm, show_images_unnorm
from ml.NeuralStyleTransfer.NST import StyleTransferModel
from ..utils import tensors_to_base64

BASIC_MODELS = [
    'DCGAN',
]
TEXT2IMAGE_MODELS = [
    'Text-DCGAN',
    'StackGAN',
]


def home(request):
    context = {}
    return render(request, 'home.html', context)


def train(request):
    context = {
        'basic_models': BASIC_MODELS,
        'text2image': TEXT2IMAGE_MODELS,
    }
    return render(request, 'train.html', context)


def inference(request):
    trained_models = [
        name for name in os.listdir(os.path.join(os.getcwd(), r'ml/resources/'))
        if os.path.isdir(os.path.join(os.getcwd(), r'ml/resources/', name))
    ]
    saved_models = {}
    for name in trained_models:
        saved_models[name] = [saved for saved in os.listdir(os.path.join(os.getcwd(), r'ml/resources/', name))
                              if os.path.isdir(os.path.join(os.getcwd(), r'ml/resources/', name, saved))]
    context = {
        'models': saved_models,
    }
    return render(request, 'inference.html', context)


def style_transfer(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # form.save()
            # Get the current instance object to display in the template
            content_image = form.cleaned_data['content_image']
            style_image = form.cleaned_data['style_image']
            image_size = int(request.POST.get('image_size'))
            alpha = int(request.POST.get('alpha'))
            beta = int(request.POST.get('beta'))
            epochs = int(request.POST.get('epochs'))
            lrn_rate = float(request.POST.get('lrn_rate1'))
            beta1 = float(request.POST.get('beta1'))
            beta2 = float(request.POST.get('beta2'))

            content_image = Image.open(content_image)
            style_image = Image.open(style_image)
            transform = transform_unnorm(image_size)
            content_image = transform(content_image).unsqueeze(0)
            style_image = transform(style_image).unsqueeze(0)

            if content_image.size(1) == 4:
                content_image = content_image[:, :3, :, :]
            if style_image.size(1) == 4:
                style_image = style_image[:, :3, :, :]

            model = StyleTransferModel(pretrained_vgg_dir=os.path.join(os.getcwd(), r'ml/resources/vgg19.pt'))
            result_image = model.mix_style(
                epochs=epochs,
                content_img=content_image,
                style_img=style_image,
                alpha=alpha,
                beta=beta,
                lrn_rate=lrn_rate,
                beta1=beta1,
                beta2=beta2,
                show_loss=False
            )

            result_image = tensors_to_base64(result_image)

            context = {
                'form': form,
                'images': result_image,
            }

            return render(request, 'style_transfer.html', context)
    else:
        form = ImageForm()
        return render(request, 'style_transfer.html', {'form': form})


def logs(request):
    log_files = [
        name for name in os.listdir(os.path.join(os.getcwd(), r'ml/logs/'))
        if os.path.isfile(os.path.join(os.getcwd(), r'ml/logs/', name))
    ]
    context = {
        'log_files': log_files,
    }
    return render(request, 'logs.html', context)


def log(request, log_file):
    return render(request, 'log.html', {'log_file': log_file})


def logging(request, log_file):
    log_file = os.path.join(os.getcwd(), r'ml/logs/', log_file)
    with open(log_file, 'r') as file:
        logs_content = file.read()
    context = {
        'logs': logs_content,
    }
    return JsonResponse(context)
