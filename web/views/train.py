from django.shortcuts import render, redirect
import os
import threading

from ml.utils.data_management import load_cub, load_data
from web.utils import load_model

# CONSTANTS
TEXT_ENCODERS = [
    "msmarco-distilbert-base-tas-b",
]
DATASETS = [
    name for name in os.listdir(os.path.join(os.getcwd(), r'datasets/'))
    if os.path.isdir(os.path.join(os.getcwd(), r'datasets/', name))
]


def train(request, model_name):
    if request.method == "POST":
        noise_channels = int(request.POST.get('noise_dim'))
        image_channels = int(request.POST.get('image_channels'))

        text_embedding_latent = int(request.POST.get('text_embedding_latent')) \
            if request.POST.get('text_embedding_latent') else None
        text_model = request.POST.get('text_model')
        image_size = 64
        batch_size = 32

        model = load_model(
            model_name,
            noise_channels=noise_channels,
            image_channels=image_channels,
            text_embedding_latent_channels=text_embedding_latent,
            text_model=text_model,
            device='cpu',
        )

        dataset_name: str = request.POST.get('dataset')
        if dataset_name.startswith('CUB'):
            dataloader, _ = load_cub(
                parent_dir=r'datasets/',
                image_size=image_size,
                batch_size=batch_size,
            )
        else:
            dataloader, _ = load_data(
                data_dir=fr'datasets/{dataset_name}',
                image_size=image_size,
                batch_size=batch_size,
            )

        epochs = int(request.POST.get('epochs'))
        lrn_rate1 = float(request.POST.get('lrn_rate1'))
        beta1_1 = float(request.POST.get('beta1_1'))
        beta2_1 = float(request.POST.get('beta2_1'))

        if model_name == 'stackgan':
            lrn_rate2 = float(request.POST.get('lrn_rate2'))
            beta1_2 = float(request.POST.get('beta1_2'))
            beta2_2 = float(request.POST.get('beta2_2'))
            training_thread = threading.Thread(target=model.train,
                                               args=(dataloader, epochs, 0, False, False, lrn_rate1, lrn_rate2,
                                                     beta1_1, beta2_1, beta1_2, beta2_2))
        else:
            training_thread = threading.Thread(target=model.train,
                                               args=(dataloader, epochs, 0, False, False, lrn_rate1, beta1_1, beta2_1))
        training_thread.daemon = True
        training_thread.start()

        return redirect('logs')

    context = {
        'text_encoders': TEXT_ENCODERS,
        'datasets': DATASETS,
    }
    return render(request, fr'train/models/{model_name}.html', context)
