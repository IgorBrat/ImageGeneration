import torch
import os
import pytest

from ml.GAN.StyleGAN import Generator, Discriminator, StyleGAN
from ml.GAN.misc.Style.AdaIN import AdaIN
from ml.GAN.misc.Style.MLP import MultiLayerPerceptron
from ml.GAN.misc.Style.ScaledLayers import WeightedScaledConvo
from ml.GAN.misc.Style.WeightedNoise import InjectWeightedNoise


class TestGenerator:
    z_channels = 256
    w_channels = 256
    in_channels = 128
    image_channels = 5
    channel_factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    gen = Generator(z_channels, w_channels, in_channels, channel_factors, image_channels)

    def test_types(self):
        isinstance(self.gen.starting_const, torch.nn.Parameter)
        isinstance(self.gen.mlp, MultiLayerPerceptron)
        isinstance(self.gen.init_adain1, AdaIN)
        isinstance(self.gen.init_adain2, AdaIN)
        isinstance(self.gen.init_noise1, InjectWeightedNoise)
        isinstance(self.gen.init_noise2, InjectWeightedNoise)
        isinstance(self.gen.init_conv, torch.nn.Conv2d)
        isinstance(self.gen.lrelu, torch.nn.LeakyReLU)

        isinstance(self.gen.init_rgb, WeightedScaledConvo)
        isinstance(self.gen.progressive_blocks, torch.nn.ModuleList)
        isinstance(self.gen.rgb_layers, torch.nn.ModuleList)

    def test_propagation(self):
        noise = torch.randn(10, self.z_channels)
        alpha = 0.5
        num_steps = 5
        assert self.gen(noise, alpha, num_steps).shape == (10, self.image_channels, 128, 128)
        num_steps = 3
        assert self.gen(noise, alpha, num_steps).shape == (10, self.image_channels, 32, 32)


class TestDiscriminator:
    in_channels = 128
    channel_factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
    image_channels = 4
    disc = Discriminator(in_channels, channel_factors, image_channels)

    def test_types(self):
        assert isinstance(self.disc.down_sample_blocks, torch.nn.ModuleList)
        assert isinstance(self.disc.rgb_layers, torch.nn.ModuleList)
        assert isinstance(self.disc.lrelu, torch.nn.LeakyReLU)
        isinstance(self.disc.init_rgb_layer, WeightedScaledConvo)
        isinstance(self.disc.avg_pool, torch.nn.AvgPool2d)
        isinstance(self.disc.final_block, torch.nn.Sequential)

    def test_propagation(self):
        noise = torch.randn(10, self.image_channels, 128, 128)
        alpha = 0.3
        num_steps = 5
        assert self.disc(noise, alpha, num_steps).shape == (10, 1)
        noise = torch.randn(8, self.image_channels, 32, 32)
        alpha = 0.3
        num_steps = 3
        assert self.disc(noise, alpha, num_steps).shape == (8, 1)


class TestStyleGAN:
    noise_channels = 256
    latent_space_dim = 256
    input_channels = 128
    output_image_channels = 3
    init_image_size = 4
    stylegan = StyleGAN(noise_channels, latent_space_dim, input_channels, output_image_channels, init_image_size)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = StyleGAN(presaved=True, save_dir='./')
        model = StyleGAN(presaved=True, save_dir=r'tests/resources/stylegan')
        assert model.noise_channels == 256
        assert model.latent_space_dim == 256
        assert model.input_channels == 128
        assert model.image_channels == 3
        assert model.init_image_size == 4
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/stylegan')
        assert isinstance(model.gen, Generator)
        assert isinstance(model.disc, Discriminator)

    def test_save(self):
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert not save_files
        self.stylegan.save(save_dir=r'tests/resources/check_save/')
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert len(save_files) == 3
        assert 'disc.pt' in save_files
        assert 'gen.pt' in save_files
        assert 'params.json' in save_files

        for file in save_files:
            os.remove(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))

    def test_generate(self):
        assert self.stylegan.generate(5, 7, 0.5).shape == (7, self.output_image_channels, 128, 128)
        assert self.stylegan.generate(3, 10, 0.5).shape == (10, self.output_image_channels, 32, 32)
        assert self.stylegan.generate(2, 15, 0.5).shape == (15, self.output_image_channels, 16, 16)
