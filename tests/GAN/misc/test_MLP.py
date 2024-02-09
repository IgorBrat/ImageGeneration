import torch
from ml.GAN.misc.Style.MLP import MultiLayerPerceptron
from ml.GAN.misc.Style.PixelNorm import PixelNormalisation
from ml.GAN.misc.Style.ScaledLayers import ScaledLinear


class TestMLP:
    z_channels = 256
    w_channels = 512
    leak = 0.4
    perceptron = MultiLayerPerceptron(z_channels, w_channels, leak)

    def test_size(self):
        assert len(self.perceptron.mlp) == 16

    def test_types(self):
        assert isinstance(self.perceptron.mlp[0], PixelNormalisation)
        assert isinstance(self.perceptron.mlp[1], ScaledLinear)
        assert isinstance(self.perceptron.mlp[2], torch.nn.LeakyReLU)
        for idx in range(3, len(self.perceptron.mlp) - 2, 2):
            assert isinstance(self.perceptron.mlp[idx], ScaledLinear)
            assert isinstance(self.perceptron.mlp[idx + 1], torch.nn.LeakyReLU)
        assert isinstance(self.perceptron.mlp[-1], ScaledLinear)

    def test_propagation(self):
        noise = torch.randn(30, self.z_channels)
        assert self.perceptron(noise).shape == (30, self.w_channels)
        noise = torch.randn(2, self.z_channels)
        assert self.perceptron(noise).shape == (2, self.w_channels)
