import os

import pytest
import torch.nn

from ml.GAN.ConditionalGAN import Generator, Discriminator, CondGAN


class TestGenerator:
    noise_channels = 120
    out_channels = 3
    gen = Generator(noise_channels, out_channels)

    def test_size(self):
        assert self.gen.noise_channels == self.noise_channels
        assert len(self.gen.model) == 5
        for idx in range(4):
            assert len(self.gen.model[idx]) == 3
        assert len(self.gen.model[-1]) == 2

    def test_types(self):
        assert isinstance(self.gen.model, torch.nn.Sequential)
        for idx in range(4):
            assert isinstance(self.gen.model[idx], torch.nn.Sequential)
            assert isinstance(self.gen.model[idx][0], torch.nn.ConvTranspose2d)
            assert isinstance(self.gen.model[idx][1], torch.nn.BatchNorm2d)
            assert isinstance(self.gen.model[idx][2], torch.nn.LeakyReLU)
        assert isinstance(self.gen.model[-1][0], torch.nn.ConvTranspose2d)
        assert isinstance(self.gen.model[-1][-1], torch.nn.Tanh)

    def test_gen_block(self):
        block = self.gen.gen_block(self.noise_channels, self.out_channels, 3, 2, 1, leaky_relu_slope=0.4)
        assert block[0].kernel_size == (3, 3)
        assert block[0].stride == (2, 2)
        assert block[0].padding == (1, 1)
        assert block[0].in_channels == self.noise_channels
        assert block[0].out_channels == self.out_channels

        assert block[1].num_features == self.out_channels

        assert block[2].negative_slope == 0.4

    def test_propagation(self):
        noise = torch.randn(10, self.noise_channels)
        result = self.gen(noise)
        assert result.shape == (10, self.out_channels, 64, 64)


class TestDiscriminator:
    image_channels = 3
    disc = Discriminator(image_channels)

    def test_size(self):
        assert len(self.disc.model) == 5
        for idx in range(4):
            assert len(self.disc.model[idx]) == 3
        assert len(self.disc.model[-1]) == 4

    def test_types(self):
        assert isinstance(self.disc.model, torch.nn.Sequential)
        for idx in range(4):
            assert isinstance(self.disc.model[idx], torch.nn.Sequential)
            assert isinstance(self.disc.model[idx][0], torch.nn.Conv2d)
            assert isinstance(self.disc.model[idx][1], torch.nn.BatchNorm2d)
            assert isinstance(self.disc.model[idx][2], torch.nn.LeakyReLU)
        assert isinstance(self.disc.model[-1][0], torch.nn.Conv2d)
        assert isinstance(self.disc.model[-1][1], torch.nn.LeakyReLU)
        assert isinstance(self.disc.model[-1][2], torch.nn.Flatten)
        assert isinstance(self.disc.model[-1][-1], torch.nn.Sigmoid)

    def test_disc_block(self):
        block = self.disc.disc_block(10, self.image_channels, 7, 3, 2, leaky_relu_slope=0.05)
        assert block[0].kernel_size == (7, 7)
        assert block[0].stride == (3, 3)
        assert block[0].padding == (2, 2)
        assert block[0].in_channels == 10
        assert block[0].out_channels == self.image_channels

        assert block[1].num_features == self.image_channels

        assert block[2].negative_slope == 0.05

    def test_propagation(self):
        images = torch.randn(15, self.image_channels, 64, 64)
        assert self.disc(images).shape == (15, 1)


class TestCondGAN:
    noise_channels = 100
    num_classes = 40
    out_channels = 3
    image_size = 64
    condgan = CondGAN(noise_channels, num_classes, out_channels)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = CondGAN(presaved=True, save_dir='./')
        model = CondGAN(presaved=True, save_dir=r'tests/resources/condgan')
        assert model.noise_channels == 100
        assert model.image_channels == 3
        assert model.classes == 40
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/condgan')
        assert isinstance(model.gen, Generator)
        assert isinstance(model.disc, Discriminator)

    def test_save(self):
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert not save_files
        self.condgan.save(save_dir=r'tests/resources/check_save/')
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
        assert (self.condgan.generate(10, 3, distribution='normal').shape ==
                (10, self.out_channels, self.image_size, self.image_size))
        assert (self.condgan.generate(100, 10, distribution='uniform').shape ==
                (100, self.out_channels, self.image_size, self.image_size))
        with pytest.raises(ValueError, match='Only "normal" and "uniform" distributions are supported for noise'):
            self.condgan.generate(5, 5, distribution='random')

    def test_prepare_for_training(self):
        self.condgan._prepare_for_training(1e-4, 0.5, 0.9)
        assert isinstance(self.condgan.gen_optim, torch.optim.Adam)
        assert isinstance(self.condgan.disc_optim, torch.optim.Adam)
