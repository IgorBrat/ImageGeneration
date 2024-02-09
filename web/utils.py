import base64
import io

from torchvision import transforms

from ml.GAN.DCGAN import DCGAN
from ml.GAN.ConditionalGAN import CondGAN
from ml.GAN.ControlGAN import ControlGAN
from ml.GAN.StyleGAN import StyleGAN
from ml.VAE.VAE import VAE
from ml.Text2Image.StackGAN import StackGAN
from ml.Text2Image.Text2Image_DCGAN import Text2ImageDCGAN


def tensors_to_base64(tensors):
    encoded_images = []
    transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToPILImage(),
    ])
    for tensor in tensors:
        tensor = tensor.clamp(0, 1)
        image = transform(tensor)

        # Convert the Pillow image to a base64-encoded string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        encoded_images.append(encoded_image)

    return encoded_images


def load_model(model_name: str, **params):
    if model_name == 'text-dcgan':
        model = Text2ImageDCGAN(
            noise_channels=params['noise_channels'],
            image_channels=params['image_channels'],
            text_embedding_latent_channels=params['text_embedding_latent_channels'],
            text_model=params['text_model'],
            device=params['device'],
        )
    elif model_name == 'stackgan':
        model = StackGAN(
            noise_channels=params['noise_channels'],
            image_channels=params['image_channels'],
            text_embedding_latent_channels=params['text_embedding_latent_channels'],
            text_model=params['text_model'],
            device=params['device'],
        )
    elif model_name == 'dcgan':
        model = DCGAN(
            noise_channels=params['noise_channels'],
            image_channels=params['image_channels'],
            device=params['device'],
        )
    elif model_name == 'condgan':
        model = CondGAN(
            noise_channels=params['noise_channels'],
            num_classes=params['num_classes'],
            image_channels=params['image_channels'],
            device=params['device'],
        )
    elif model_name == 'controlgan':
        model = ControlGAN(
            noise_channels=params['noise_channels'],
            features=params['features'],
            image_channels=params['image_channels'],
            device=params['device'],
        )
    elif model_name == 'stylegan':
        model = StyleGAN(
            noise_channels=params['noise_channels'],
            latent_space_dim=params['latent_space_dim'],
            input_channels=params['input_channels'],
            output_image_channels=params['output_channels'],
            init_image_size=params['init_image_size'],
            device=params['device'],
        )
    elif model_name == 'vae':
        model = VAE(
            image_size=params['image_size'],
            image_channels=params['image_channels'],
            device=params['device'],
        )
    else:
        raise ValueError(f'Model {model_name} is not supported. Check for correct spelling and supported models.')
    return model


def load_presaved_model(model_name: str, version: str):
    if model_name == 'text-dcgan':
        model = Text2ImageDCGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\text-dcgan\\{version}',
        )
    elif model_name == 'stackgan':
        model = StackGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\stackgan\\{version}',
        )
    elif model_name == 'dcgan':
        model = DCGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\dcgan\\{version}',
        )
    elif model_name == 'condgan':
        model = CondGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\condgan\\{version}',
        )
    elif model_name == 'controlgan':
        model = ControlGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\controlgan\\{version}',
        )
    elif model_name == 'stylegan':
        model = StyleGAN(
            presaved=True,
            save_dir=fr'ml\\resources\\stylegan\\{version}',
        )
    elif model_name == 'vae':
        model = VAE(
            presaved=True,
            save_dir=fr'ml\\resources\\vae\\{version}',
        )
    else:
        raise ValueError(f'Model {model_name} is not supported. Check for correct spelling and supported models.')
    return model
