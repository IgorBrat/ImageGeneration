import torch

from ml.GAN.misc.Style.WeightedNoise import InjectWeightedNoise


def test_weighted_noise():
    in_channels = 5
    layer = InjectWeightedNoise(in_channels)
    features = torch.randn(15, in_channels, 16, 16)
    assert layer(features).shape == (15, in_channels, 16, 16)
    features = torch.randn(7, in_channels, 32, 32)
    assert layer(features).shape == (7, in_channels, 32, 32)
