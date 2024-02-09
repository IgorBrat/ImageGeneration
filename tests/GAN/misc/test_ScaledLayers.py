import torch

from ml.GAN.misc.Style.ScaledLayers import ScaledLinear, WeightedScaledConvo


def test_scaled_linear():
    in_features = 256
    out_features = 128
    layer = ScaledLinear(in_features, out_features)
    assert layer.scale == (2 / in_features) ** 0.5
    features = torch.randn(15, in_features)
    assert layer(features).shape == (15, out_features)


def test_scaled_convolution():
    in_features = 128
    out_features = 512
    kernel = 5
    stride = 3
    padding = 1
    convo = WeightedScaledConvo(in_features, out_features, kernel, stride, padding)
    assert convo.scale == (2 / (in_features * kernel ** 2)) ** 0.5
    assert convo.layer.kernel_size == (kernel, kernel)
    assert convo.layer.stride == (stride, stride)
    assert convo.layer.padding == (padding, padding)
    features = torch.randn(10, in_features, 8, 8)
    assert convo(features).shape == (10, out_features, 2, 2)

