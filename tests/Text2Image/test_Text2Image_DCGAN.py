import pytest
import torch
import os

from ml.Text2Image.Text2Image_DCGAN import Generator, Discriminator, Text2ImageDCGAN


class TestGenerator:
    noise_channels = 128
    out_channels = 3
    text_embedding_channels = 256
    text_embedding_latent = 128
    gen = Generator(noise_channels, out_channels, text_embedding_channels, text_embedding_latent)

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
        result = self.gen(noise, embedding)
        assert result.shape == (10, self.out_channels, 64, 64)


class TestDiscriminator:
    image_channels = 3
    text_embedding_channels = 250
    text_embedding_latent = 128
    disc = Discriminator(image_channels, text_embedding_channels, text_embedding_latent)

    def test_size(self):
        assert len(self.disc.model) == 4
        for idx in range(4):
            assert len(self.disc.model[idx]) == 3
        assert len(self.disc.out) == 4
        assert len(self.disc.text_embed) == 3

    def test_types(self):
        assert isinstance(self.disc.model, torch.nn.Sequential)
        for idx in range(4):
            assert isinstance(self.disc.model[idx], torch.nn.Sequential)
            assert isinstance(self.disc.model[idx][0], torch.nn.Conv2d)
            assert isinstance(self.disc.model[idx][1], torch.nn.BatchNorm2d)
            assert isinstance(self.disc.model[idx][2], torch.nn.LeakyReLU)
        assert isinstance(self.disc.text_embed[0], torch.nn.Linear)
        assert isinstance(self.disc.text_embed[1], torch.nn.BatchNorm1d)
        assert isinstance(self.disc.text_embed[2], torch.nn.LeakyReLU)

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
        images = torch.randn(20, self.image_channels, 64, 64)
        embedding = torch.randn(20, self.text_embedding_channels)
        out, cnn_out = self.disc(images, embedding)
        assert out.shape == (20, 1)
        assert cnn_out.shape == (20, 256, 4, 4)


class TestText2ImageDCGAN:
    noise_channels = 100
    image_channels = 3
    text_embedding_latent = 100
    text_model = "msmarco-distilbert-base-tas-b"
    image_size = 64
    model = Text2ImageDCGAN(noise_channels, image_channels, text_embedding_latent, text_model)

    def test_load(self):
        with pytest.raises(FileNotFoundError):
            model = Text2ImageDCGAN(presaved=True, save_dir='./')
        model = Text2ImageDCGAN(presaved=True, save_dir=r'tests/resources/text_dcgan')
        assert model.noise_channels == self.noise_channels
        assert model.image_channels == self.image_channels
        with pytest.raises(FileNotFoundError):
            model.load('./')
        model.load(r'tests/resources/text_dcgan')
        assert isinstance(model.gen, Generator)
        assert isinstance(model.disc, Discriminator)

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
        assert len(save_files) == 3
        assert 'disc.pt' in save_files
        assert 'gen.pt' in save_files
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
        self.model._prepare_for_training(1e-3, 0.7, 0.99)
        assert isinstance(self.model.gen_optim, torch.optim.Adam)
        assert isinstance(self.model.disc_optim, torch.optim.Adam)
