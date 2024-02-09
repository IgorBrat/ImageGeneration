import torch

from ml.GAN.misc.Style.AdaIN import AdaIN
from ml.GAN.misc.Style.Blocks import GeneratorBlock, Conv2Block
from ml.GAN.misc.Style.ScaledLayers import WeightedScaledConvo
from ml.GAN.misc.Style.WeightedNoise import InjectWeightedNoise


class TestGeneratorBlock:
    in_channels = 32
    out_channels = 128
    w_channels = 256
    block = GeneratorBlock(in_channels, out_channels, w_channels)

    def test_types(self):
        isinstance(self.block.convo1, WeightedScaledConvo)
        isinstance(self.block.convo2, WeightedScaledConvo)
        isinstance(self.block.lrelu, torch.nn.LeakyReLU)
        isinstance(self.block.noise_inject1, InjectWeightedNoise)
        isinstance(self.block.noise_inject2, InjectWeightedNoise)
        isinstance(self.block.adain1, AdaIN)
        isinstance(self.block.adain2, AdaIN)

    def test_propagation(self):
        features = torch.randn(10, self.in_channels, 4, 4)
        w_noise = torch.randn(10, self.w_channels)
        assert self.block(features, w_noise).shape == (10, self.out_channels, 4, 4)
        features = torch.randn(3, self.in_channels, 32, 32)
        w_noise = torch.randn(3, self.w_channels)
        assert self.block(features, w_noise).shape == (3, self.out_channels, 32, 32)


class TestConv2Block:
    in_channels = 8
    out_channels = 64
    leak = 0.5
    block = Conv2Block(in_channels, out_channels)

    def test_types(self):
        isinstance(self.block.convo1, WeightedScaledConvo)
        isinstance(self.block.convo2, WeightedScaledConvo)
        isinstance(self.block.lrelu, torch.nn.LeakyReLU)

    def test_propagation(self):
        features = torch.randn(15, self.in_channels, 8, 8)
        assert self.block(features).shape == (15, self.out_channels, 8, 8)
        features = torch.randn(5, self.in_channels, 64, 64)
        assert self.block(features).shape == (5, self.out_channels, 64, 64)
