import numpy as np
import torch
import pytest
import os
from ml.VAE.VAE import get_vae_loss, VaeNet, VAE


def test_vae_loss():
    x = torch.Tensor([[0.6]])
    y = torch.Tensor([[0.9]])
    mean = torch.Tensor([3])
    log_variance = torch.Tensor([5])
    assert np.isclose(get_vae_loss(x, y, mean, log_variance, (1, 1)), 76.2579)
    x = torch.Tensor([[0.1]])
    y = torch.Tensor([[0.95]])
    mean = torch.Tensor([10])
    log_variance = torch.Tensor([7])
    assert np.isclose(get_vae_loss(x, y, mean, log_variance, (1, 1)), 596.5093)


class TestVaeNet:
    size = 64
    channels = 3
    vae_net = VaeNet(size, channels)

    def test_types(self):
        isinstance(self.vae_net.encoder_input, torch.nn.Linear)
        isinstance(self.vae_net.batch_norm_input, torch.nn.BatchNorm1d)
        isinstance(self.vae_net.encoder_mean, torch.nn.Linear)
        isinstance(self.vae_net.encoder_log_variance, torch.nn.Linear)
        isinstance(self.vae_net.decoder, torch.nn.Sequential)
        assert len(self.vae_net.decoder) == 5
        isinstance(self.vae_net.decoder[0], torch.nn.Linear)
        isinstance(self.vae_net.decoder[1], torch.nn.BatchNorm1d)
        isinstance(self.vae_net.decoder[2], torch.nn.LeakyReLU)
        isinstance(self.vae_net.decoder[3], torch.nn.Linear)
        isinstance(self.vae_net.decoder[-1], torch.nn.Sigmoid)

    def test_encode(self):
        images = torch.randn(15, self.channels, self.size, self.size)
        mean, log_var = self.vae_net.encode(images.view(-1, self.channels * self.size ** 2))
        assert mean.shape == (15, self.size)
        assert mean.shape == (15, self.size)

    def test_decode(self):
        latent_noise = torch.randn(10, self.size)
        assert self.vae_net.decode(latent_noise).shape == (10, self.channels * self.size ** 2)
        latent_noise = torch.randn(22, self.size)
        assert self.vae_net.decode(latent_noise).shape == (22, self.channels * self.size ** 2)

    def test_propagation(self):
        images = torch.randn(10, self.channels, self.size, self.size)
        vae_images, _, _ = self.vae_net(images)
        assert vae_images.shape == (10, self.channels * self.size * self.size)


class TestVAE:
    image_size = 64
    image_channels = 3
    vae = VAE(image_size, image_channels)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = VAE(presaved=True, save_dir='./')
        model = VAE(presaved=True, save_dir=r'tests/resources/vae')
        assert model.vae_net.size == self.image_size
        assert model.vae_net.channels == self.image_channels
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/vae')
        assert isinstance(model.vae_net, VaeNet)

    def test_save(self):
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert not save_files
        self.vae.save(save_dir=r'tests/resources/check_save/')
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert len(save_files) == 2
        assert 'vae.pt' in save_files
        assert 'params.json' in save_files

        for file in save_files:
            os.remove(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))

    def test_generate(self):
        assert (self.vae.generate(10, distribution='normal').shape ==
                (10, self.image_channels, self.image_size, self.image_size))
        assert (self.vae.generate(100, distribution='uniform').shape ==
                (100, self.image_channels, self.image_size, self.image_size))
        with pytest.raises(ValueError, match='Only "normal" and "uniform" distributions are supported for noise'):
            self.vae.generate(5, distribution='random')

    def test_prepare_for_training(self):
        self.vae._prepare_for_training(1e-3, 0.9, 0.95)
        assert isinstance(self.vae.optim, torch.optim.Adam)
