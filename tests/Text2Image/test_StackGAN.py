import pytest
import torch
import os

from ml.Text2Image.StackGAN import (Stage1Gen, Stage1Disc, Stage2Gen, Stage2Disc, Stage1GAN, Stage2GAN, StackGAN)
from ml.utils.text_processing import ConditionAugmentation


class TestStage1Gen:
    noise_channels = 128
    out_channels = 3
    text_embedding_channels = 256
    text_embedding_latent = 192
    gen = Stage1Gen(noise_channels, out_channels, text_embedding_channels, text_embedding_latent)

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
        embedding = torch.randn(10, self.text_embedding_channels)
        images, mean, log_var = self.gen(noise, embedding)
        assert images.shape == (10, self.out_channels, 64, 64)
        assert mean.shape == (10, self.text_embedding_latent)
        assert log_var.shape == (10, self.text_embedding_latent)


class TestStage1Disc:
    image_channels = 3
    text_embedding_channels = 256
    text_embedding_latent = 144
    disc = Stage1Disc(image_channels, text_embedding_channels, text_embedding_latent)

    def test_size(self):
        assert len(self.disc.model) == 4
        for idx in range(4):
            assert len(self.disc.model[idx]) == 3
        assert len(self.disc.out) == 4

    def test_types(self):
        assert isinstance(self.disc.model, torch.nn.Sequential)
        for idx in range(4):
            assert isinstance(self.disc.model[idx], torch.nn.Sequential)
            assert isinstance(self.disc.model[idx][0], torch.nn.Conv2d)
            assert isinstance(self.disc.model[idx][1], torch.nn.BatchNorm2d)
            assert isinstance(self.disc.model[idx][2], torch.nn.LeakyReLU)
        assert isinstance(self.disc.cond_aug, ConditionAugmentation)

        assert isinstance(self.disc.out[0], torch.nn.Conv2d)
        assert isinstance(self.disc.out[1], torch.nn.LeakyReLU)
        assert isinstance(self.disc.out[2], torch.nn.Flatten)
        assert isinstance(self.disc.out[-1], torch.nn.Sigmoid)

    def test_disc_block(self):
        block = self.disc.disc_block(10, self.image_channels, 7, 3, 2, leaky_relu_slope=0.1)
        assert block[0].kernel_size == (7, 7)
        assert block[0].stride == (3, 3)
        assert block[0].padding == (2, 2)
        assert block[0].in_channels == 10
        assert block[0].out_channels == self.image_channels

        assert block[1].num_features == self.image_channels

        assert block[2].negative_slope == 0.1

    def test_propagation(self):
        images = torch.randn(15, self.image_channels, 64, 64)
        embedding = torch.randn(15, self.text_embedding_channels)
        out, cnn_out = self.disc(images, embedding)
        assert out.shape == (15, 1)
        assert cnn_out.shape == (15, 256, 4, 4)


class TestStage2Gen:
    input_channels = 100
    out_channels = 3
    text_embedding_channels = 256
    text_embedding_latent = 128
    gen = Stage2Gen(input_channels, out_channels, text_embedding_channels, text_embedding_latent)

    def test_size(self):
        assert self.gen.input_channels == self.input_channels
        assert len(self.gen.down_sample) == 2
        for idx in range(2):
            assert len(self.gen.down_sample[idx]) == 3

        assert len(self.gen.up_sample) == 10
        assert len(self.gen.up_sample[-1]) == 2

    def test_types(self):
        assert isinstance(self.gen.down_sample, torch.nn.Sequential)
        for idx in range(0, 7, 3):
            assert isinstance(self.gen.up_sample[idx], torch.nn.Sequential)
            assert isinstance(self.gen.up_sample[idx][0], torch.nn.ConvTranspose2d)
            assert isinstance(self.gen.up_sample[idx][1], torch.nn.BatchNorm2d)
            assert isinstance(self.gen.up_sample[idx][2], torch.nn.LeakyReLU)

        assert isinstance(self.gen.up_sample[-1], torch.nn.Sequential)
        assert isinstance(self.gen.up_sample[-1][0], torch.nn.ConvTranspose2d)
        assert isinstance(self.gen.up_sample[-1][-1], torch.nn.Tanh)

    def test_conv_block(self):
        block = self.gen.conv_block(self.input_channels, self.out_channels, 3, 2, 1, leak=0.4)
        assert block[0].kernel_size == (3, 3)
        assert block[0].stride == (2, 2)
        assert block[0].padding == (1, 1)
        assert block[0].in_channels == self.input_channels
        assert block[0].out_channels == self.out_channels

        assert block[1].num_features == self.out_channels

        assert block[2].negative_slope == 0.4

    def test_deconv_block(self):
        block = self.gen.deconv_block(self.input_channels, self.out_channels, 5, 0, 2, leak=0.25)
        assert block[0].kernel_size == (5, 5)
        assert block[0].stride == (0, 0)
        assert block[0].padding == (2, 2)
        assert block[0].in_channels == self.input_channels
        assert block[0].out_channels == self.out_channels

        assert block[1].num_features == self.out_channels

        assert block[2].negative_slope == 0.25

    def test_propagation(self):
        noise = torch.randn(10, self.input_channels, 64, 64)
        embedding = torch.randn(10, self.text_embedding_channels)
        images, mean, log_var = self.gen(noise, embedding)
        assert images.shape == (10, self.out_channels, 256, 256)
        assert mean.shape == (10, self.text_embedding_latent)
        assert log_var.shape == (10, self.text_embedding_latent)


class TestStage2Disc:
    image_channels = 3
    text_embedding_channels = 256
    text_embedding_latent = 128
    disc = Stage2Disc(image_channels, text_embedding_channels, text_embedding_latent)

    def test_size(self):
        assert len(self.disc.down_sample) == 23
        assert len(self.disc.out) == 4

    def test_types(self):
        assert isinstance(self.disc.down_sample, torch.nn.Sequential)
        assert isinstance(self.disc.cond_aug, ConditionAugmentation)

        assert isinstance(self.disc.out[0], torch.nn.Flatten)
        assert isinstance(self.disc.out[1], torch.nn.Linear)
        assert isinstance(self.disc.out[2], torch.nn.Linear)
        assert isinstance(self.disc.out[-1], torch.nn.Sigmoid)

    def test_pooling_block(self):
        block = self.disc.pooling_block(2, 1, 0, leak=0.15)
        assert block[0].kernel_size == 2
        assert block[0].stride == 1
        assert block[0].padding == 0

        assert block[1].negative_slope == 0.15

    def test_propagation(self):
        images = torch.randn(5, self.image_channels, 256, 256)
        embedding = torch.randn(5, self.text_embedding_channels)
        out = self.disc(images, embedding)
        assert out.shape == (5, 1)


class TestStackGAN:
    noise_channels = 100
    image_channels = 3
    text_embedding_latent = 128
    image_size = 256
    model = StackGAN(noise_channels, image_channels, text_embedding_latent)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = StackGAN(presaved=True, save_dir='./')
        model = StackGAN(presaved=True, save_dir=r'tests/resources/stackgan')
        assert model.noise_channels == self.noise_channels
        assert model.image_channels == self.image_channels
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/stackgan')
        assert isinstance(model.stage1gan, Stage1GAN)
        assert isinstance(model.stage2gan, Stage2GAN)
        assert isinstance(model.stage1gan.gen, Stage1Gen)
        assert isinstance(model.stage1gan.disc, Stage1Disc)
        assert isinstance(model.stage2gan.gen, Stage2Gen)
        assert isinstance(model.stage2gan.disc, Stage2Disc)

    def test_save(self):
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert not save_files
        self.model.save(save_dir=r'tests/resources/check_save/')
        save_files = [
            file for file in os.listdir(os.path.join(os.getcwd(), r'tests/resources/check_save/'))
            if os.path.isfile(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))
        ]
        assert len(save_files) == 5
        assert 'stage1gen.pt' in save_files
        assert 'stage1disc.pt' in save_files
        assert 'stage2gen.pt' in save_files
        assert 'stage2disc.pt' in save_files
        assert 'params.json' in save_files

        for file in save_files:
            os.remove(os.path.join(os.getcwd(), r'tests/resources/check_save/', file))

    def test_generate(self):
        assert (self.model.generate('red beautiful bird', 10, distribution='normal').shape ==
                (10, self.image_channels, self.image_size, self.image_size))
        assert (self.model.generate('fastest cat in the world', 20, distribution='uniform').shape ==
                (20, self.image_channels, self.image_size, self.image_size))
        with pytest.raises(ValueError, match='Only "normal" and "uniform" distributions are supported for noise'):
            self.model.generate('cute dog', 5, distribution='random')

    def test_prepare_for_training(self):
        self.model._prepare_for_training(1e-3, 1e-4, 0.9, 0.99, 0.8, 0.9)
        assert isinstance(self.model.stage1gan.gen_optim, torch.optim.Adam)
        assert isinstance(self.model.stage1gan.disc_optim, torch.optim.Adam)
        assert isinstance(self.model.stage2gan.gen_optim, torch.optim.Adam)
        assert isinstance(self.model.stage2gan.disc_optim, torch.optim.Adam)
