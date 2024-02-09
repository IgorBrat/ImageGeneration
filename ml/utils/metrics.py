from typing import Optional

from torchvision.models import inception_v3
import torch
import numpy as np
from torchvision.transforms import transforms
from tqdm.auto import tqdm
from ml.utils.tensor_logic import get_normal_noise, matrix_sqrt

import lpips


# region Fréchet Inception Distance

def get_inception_model_(path: Optional[str] = None, device: str = 'cpu'):
    if path:
        inception_model = inception_v3(init_weights=False, weights=None)
        inception_model.load_state_dict(torch.load(path)['model_state_dict'])
    else:
        inception_model = inception_v3(pretrained=True)
    inception_model.fc = torch.nn.Identity()
    inception_model.to(device)
    return inception_model.eval()


def preprocess_image_(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img


def get_covariance_(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def frechet_inception_distance(mean_real, mean_fake, std_real, std_fake):
    return torch.norm(mean_fake - mean_real) ** 2 + torch.trace(
        std_real + std_fake - 2 * matrix_sqrt(std_real @ std_fake))


def get_fid(gen, dataloader, samples, batch_size,
            noise_dim, inception_model_path: Optional[str] = None, device: str = 'cpu'):
    """
    Calculate Fréchet Inception Distance
    :param gen: generator model
    :param dataloader: Dataloader object
    :param samples: number of samples to calculate FID on
    :param batch_size: dataset images batch
    :param noise_dim: channels of input noise vector
    :param inception_model_path: path to InceptionV3 model
    :param device: device to move images to
    :return: FID
    """
    inception_model = get_inception_model_(inception_model_path)
    # InceptionV3 expects images of size 299x299
    resize = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
    ])

    fake_features_list = []
    real_features_list = []

    gen.eval()

    cur_samples = 0
    with torch.no_grad():
        for real_example, *_ in tqdm(dataloader, total=samples // batch_size):  # Go by batch
            real_samples = resize(real_example)
            real_features = inception_model(real_samples.to(device)).detach().to('cpu')  # Move features to CPU
            real_features_list.append(real_features)

            fake_samples = get_normal_noise((len(real_example), noise_dim), device)
            fake_samples = preprocess_image_(gen(fake_samples))
            fake_features = inception_model(fake_samples.to(device)).detach().to('cpu')
            fake_features_list.append(fake_features)
            cur_samples += len(real_samples)
            if cur_samples > samples:
                break
    fake_features_all = torch.cat(fake_features_list)
    real_features_all = torch.cat(real_features_list)

    mean_fake = torch.mean(fake_features_all, dim=0)
    mean_real = torch.mean(real_features_all, dim=0)
    std_fake = get_covariance_(fake_features_all)
    std_real = get_covariance_(real_features_all)

    with torch.no_grad():
        return frechet_inception_distance(mean_real, mean_fake, std_real, std_fake).item()


# endregion

# region Perceptual Path Length


def get_ppl_w(gen, noise_dim, samples, model_path: str, eps: float = 1e-4):
    """
    Calculate Perceptual Path Length in latent w-space
    :param gen: generator model
    :param noise_dim: channels of input noise vector
    :param samples: number of samples to calculate PPL on
    :param model_path: path to VGG model
    :param eps: interpolation step
    :return: PPL in latent w-space
    """
    # Perceptual Path Length in w-space (latent)
    ppl_loss = lpips.LPIPS(net='vgg', model_path=model_path)
    gen.eval()

    # Sample two points in w-space
    map_net = torch.nn.Identity()
    w_1 = map_net(torch.randn(samples, noise_dim))
    w_2 = map_net(torch.randn(samples, noise_dim))
    # Sample num_samples points along the interpolated lines
    t = torch.rand(samples)[:, None]
    # Interpolate between the points
    interpolated_1 = torch.lerp(w_1, w_2, t)
    interpolated_2 = torch.lerp(w_1, w_2, t + eps)
    # Generated the interpolated images
    y_1, y_2 = gen(interpolated_1), gen(interpolated_2)
    # Calculate the per-sample LPIPS
    cur_lpips = ppl_loss(y_1, y_2)
    # Calculate the PPL from the LPIPS
    ppl = cur_lpips / (eps ** 2)
    return ppl.mean().item()


def normalize_(x):
    return x / torch.norm(x, dim=1)[:, None]


def get_omega_(x, y):
    return torch.acos((normalize_(x) * normalize_(y)).sum(1))


def slerp_(x, y, t):
    omega = get_omega_(x, y)[:, None]
    c1 = torch.sin(omega * (1 - t)) / torch.sin(omega)
    c2 = torch.sin(omega * t) / torch.sin(omega)
    return c1 * x + c2 * y


def get_ppl_z(gen, noise_dim, samples, model_path: str, device: str, eps: float = 1e-4):
    """
    Calculate Perceptual Path Length in input noise z-space
    :param gen: generator model
    :param noise_dim: channels of input noise vector
    :param samples: number of samples to calculate PPL on
    :param model_path: path to VGG model
    :param device: device to move images to
    :param eps: interpolation step
    :return: PPL in input noise z-space
    """
    # Perceptual Path Length in spherical z-space (initial noise)
    ppl_loss = lpips.LPIPS(net='vgg', model_path=model_path)
    gen.eval()

    # Sample of a batch of num_samples pairs of points
    z_1 = torch.randn(samples, noise_dim).to(device)
    z_2 = torch.randn(samples, noise_dim).to(device)
    # Sample num_samples points along the interpolated lines
    t = torch.rand(samples)[:, None]
    # Interpolate between the points
    interpolated_1 = slerp_(z_1, z_2, t)
    interpolated_2 = slerp_(z_1, z_2, t + eps)
    # Generated the interpolated images
    y_1, y_2 = gen(interpolated_1), gen(interpolated_2)
    # Calculate the per-sample LPIPS
    cur_lpips = ppl_loss(y_1, y_2)
    # Calculate the PPL from the LPIPS
    ppl = cur_lpips / (eps ** 2)
    return ppl

# endregion
