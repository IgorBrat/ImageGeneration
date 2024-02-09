from ml.utils.network_misc import count_parameters, init_weights, ResBlock
import torch
import numpy as np


class TestCountParameters:

    def test_empty_model(self):
        model = torch.nn.Sequential()
        train_params_count, static_params_count = count_parameters(model)
        assert not train_params_count
        assert not static_params_count

    def test_conv_model(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(100, 200, 3, 1, 1),
            torch.nn.BatchNorm2d(200),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(200, 200, 3, 1, 1),
            torch.nn.BatchNorm2d(200),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(2, 100),
            torch.nn.Flatten(),
            torch.nn.Sigmoid(),
        )
        train_params_count, static_params_count = count_parameters(model)
        assert train_params_count == 541500
        assert static_params_count == 0


class TestInitWeights:
    def test_conv_layer(self):
        layer = torch.nn.Conv2d(100, 30, 3, 1, 0)
        init_weights(layer)
        assert np.isclose(torch.mean(layer.weight, dim=(0, 1, 2, 3)).detach().numpy(), 0, atol=1e-3, rtol=0)
        assert np.isclose(torch.std(layer.weight, dim=(0, 1, 2, 3)).detach().numpy(), 0.02, atol=1e-3, rtol=0)

    def test_batch_layer(self):
        layer = torch.nn.BatchNorm2d(1000)
        init_weights(layer)
        assert np.isclose(torch.mean(layer.weight).detach().numpy(), 1, atol=1e-2, rtol=0)
        assert np.isclose(torch.std(layer.weight).detach().numpy(), 0.02, atol=1e-2, rtol=0)
        assert np.allclose(layer.bias.detach().numpy(), torch.zeros_like(layer.bias), atol=1e-5, rtol=0)


class TestResBlock:
    channels = 100
    res_block = ResBlock(channels, channels)

    def test_size(self):
        assert self.res_block.size() == 5

    def test_types(self):
        assert isinstance(self.res_block.block[0], torch.nn.ConvTranspose2d)
        assert isinstance(self.res_block.block[1], torch.nn.BatchNorm2d)
        assert isinstance(self.res_block.block[2], torch.nn.ConvTranspose2d)
        assert isinstance(self.res_block.block[3], torch.nn.BatchNorm2d)
        assert isinstance(self.res_block.block[4], torch.nn.LeakyReLU)

    def test_result_shape(self):
        shape1 = (10, self.channels, 64, 64)
        shape2 = (5, self.channels, 128, 128)
        noise1 = torch.randn(shape1)
        noise2 = torch.randn(shape2)
        assert self.res_block(noise1).shape == shape1
        assert self.res_block(noise2).shape == shape2
