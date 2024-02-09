import os

import pytest
import torch
import numpy as np

from ml.GAN.ControlGAN import Generator, Discriminator, Classifier, ControlGAN
from ml.utils.network_misc import ResBlock


class TestGenerator:
    noise_channels = 200
    out_channels = 5
    gen = Generator(noise_channels, out_channels)

    def test_size(self):
        assert self.gen.input_channels == self.noise_channels
        assert len(self.gen.model) == 12
        assert len(self.gen.model[3]) == 3
        assert len(self.gen.model[-1]) == 2

    def test_types(self):
        assert isinstance(self.gen.model, torch.nn.Sequential)
        assert isinstance(self.gen.model[1], ResBlock)
        assert isinstance(self.gen.model[8][0], torch.nn.ConvTranspose2d)
        assert isinstance(self.gen.model[8][1], torch.nn.BatchNorm2d)
        assert isinstance(self.gen.model[8][2], torch.nn.LeakyReLU)
        assert isinstance(self.gen.model[-1][0], torch.nn.ConvTranspose2d)
        assert isinstance(self.gen.model[-1][-1], torch.nn.Tanh)

    def test_gen_block(self):
        block = self.gen.deconv_block(self.noise_channels, self.out_channels, 3, 2, 3, leak=0.5)
        assert block[0].kernel_size == (3, 3)
        assert block[0].stride == (2, 2)
        assert block[0].padding == (3, 3)
        assert block[0].in_channels == self.noise_channels
        assert block[0].out_channels == self.out_channels

        assert block[1].num_features == self.out_channels

        assert block[2].negative_slope == 0.5

    def test_propagation(self):
        noise_size = 32
        noise = torch.randn(25, self.noise_channels, noise_size, noise_size)
        result = self.gen(noise)
        assert result.shape == (25, self.out_channels, noise_size * 4, noise_size * 4)
        noise = torch.randn(50, self.noise_channels, noise_size // 2, noise_size // 2)
        result = self.gen(noise)
        assert result.shape == (50, self.out_channels, noise_size * 2, noise_size * 2)


class TestDiscriminator:
    image_channels = 3
    disc = Discriminator(image_channels)

    def test_size(self):
        assert len(self.disc.model) == 20
        assert len(self.disc.model[3]) == 2
        assert len(self.disc.model[8]) == 2
        assert len(self.disc.model[13]) == 2

    def test_types(self):
        assert isinstance(self.disc.model, torch.nn.Sequential)
        assert isinstance(self.disc.model[1], ResBlock)

        assert isinstance(self.disc.model[3], torch.nn.Sequential)
        assert isinstance(self.disc.model[3][0], torch.nn.AvgPool2d)
        assert isinstance(self.disc.model[3][1], torch.nn.LeakyReLU)

        assert isinstance(self.disc.model[-6], torch.nn.BatchNorm2d)
        assert isinstance(self.disc.model[-5], torch.nn.LeakyReLU)
        assert isinstance(self.disc.model[-4], torch.nn.Flatten)
        assert isinstance(self.disc.model[-2], torch.nn.Linear)
        assert isinstance(self.disc.model[-1], torch.nn.Sigmoid)

    def test_pooling_block(self):
        block = self.disc.pooling_block(5, 2, 3, 0.4)
        assert block[0].kernel_size == 5
        assert block[0].stride == 2
        assert block[0].padding == 3

        assert block[1].negative_slope == 0.4

    def test_propagation(self):
        images = torch.randn(13, self.image_channels, 128, 128)
        assert self.disc(images).shape == (13, 1)

        images = torch.randn(24, self.image_channels, 128, 128)
        assert self.disc(images).shape == (24, 1)


class TestControlGAN:
    noise_channels = 100
    image_channels = 3
    features = 30
    image_size = 128
    control_gan = ControlGAN(noise_channels, features, image_channels)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = ControlGAN(presaved=True, save_dir='./')
        model = ControlGAN(presaved=True, save_dir=r'tests/resources/controlgan')
        assert model.noise_channels == self.noise_channels
        assert model.image_channels == self.image_channels
        assert model.features == self.features
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/controlgan')
        assert isinstance(model.gen, Generator)
        assert isinstance(model.disc, Discriminator)
        assert isinstance(model.classif, Classifier)

    def test_save(self):
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert not save_files
        self.control_gan.save(save_dir=r'tests/resources/check_save/')
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert len(save_files) == 4
        assert 'disc.pt' in save_files
        assert 'gen.pt' in save_files
        assert 'classif.pt' in save_files
        assert 'params.json' in save_files

        for file in save_files:
            os.remove(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))

    def test_disc_loss(self):
        crit = torch.nn.BCELoss()
        fake = torch.Tensor([0.3])
        real = torch.Tensor([0.9])
        with pytest.raises(ValueError, match=r'Alpha parameter for discriminator loss must be in range \[0, 1\]'):
            self.control_gan.get_disc_loss(crit, real, fake, alpha=1.4)
        with pytest.raises(ValueError, match=r'Alpha parameter for discriminator loss must be in range \[0, 1\]'):
            self.control_gan.get_disc_loss(crit, real, fake, alpha=-2)
        assert np.isclose(
            self.control_gan.get_disc_loss(crit, real, fake, alpha=1),
            -1 * np.log(real),
            atol=1e-6,
            rtol=0,
        )
        assert np.isclose(
            self.control_gan.get_disc_loss(crit, real, fake, alpha=0),
            -1 * np.log(1 - fake),
            atol=1e-6,
            rtol=0,
        )
        assert np.isclose(
            self.control_gan.get_disc_loss(crit, real, fake, alpha=0.3),
            0.3 * (-1 * np.log(real)) + 0.7 * (-1 * np.log(1 - fake)),
            atol=1e-6,
            rtol=0,
        )
        assert np.isclose(
            self.control_gan.get_disc_loss(crit, real, fake, alpha=0.9),
            0.9 * (-1 * np.log(real)) + 0.1 * (-1 * np.log(1 - fake)),
            atol=1e-6,
            rtol=0,
        )

    def test_gen_loss(self):
        crit = torch.nn.BCELoss()
        disc_pred = torch.Tensor([0.15])
        classif_pred = torch.Tensor([0.6])
        real = torch.Tensor([0.95])
        gen_loss, classif_loss = self.control_gan.get_gen_loss(crit, classif_pred, disc_pred, real, gamma=0)
        assert np.isclose(
            gen_loss,
            -1 * np.log(disc_pred),
            atol=1e-6,
            rtol=0,
        )
        assert np.isclose(
            classif_loss,
            -real * np.log(classif_pred) - (1 - real) * (np.log(1 - classif_pred)),
            atol=1e-6,
            rtol=0,
        )

        gen_loss, classif_loss = self.control_gan.get_gen_loss(crit, classif_pred, disc_pred, real, gamma=2)
        assert np.isclose(
            gen_loss,
            2 * (-real * np.log(classif_pred) - (1 - real) * (np.log(1 - classif_pred))) - 1 * np.log(disc_pred),
            atol=1e-6,
            rtol=0,
        )
        assert np.isclose(
            classif_loss,
            -real * np.log(classif_pred) - (1 - real) * (np.log(1 - classif_pred)),
            atol=1e-6,
            rtol=0,
        )

    def test_generate(self):
        assert (self.control_gan.generate(15, feature=3, distribution='normal').shape ==
                (15, self.image_channels, self.image_size, self.image_size))
        assert (self.control_gan.generate(42, feature=15, distribution='uniform').shape ==
                (42, self.image_channels, self.image_size, self.image_size))
        with pytest.raises(ValueError, match='Only "normal" and "uniform" distributions are supported for noise'):
            self.control_gan.generate(5, feature=1, distribution='random')

    def test_prepare_for_training(self):
        self.control_gan._prepare_for_training(1e-2, 0.3, 0.92)
        assert isinstance(self.control_gan.gen_optim, torch.optim.Adam)
        assert isinstance(self.control_gan.disc_optim, torch.optim.Adam)
        assert not self.control_gan.classif_optim
