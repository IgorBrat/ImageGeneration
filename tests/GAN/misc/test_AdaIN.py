import torch.nn

from ml.GAN.misc.Style.AdaIN import AdaIN
from ml.GAN.misc.Style.ScaledLayers import ScaledLinear


class TestAdaIN:
    adain = AdaIN(256, 512)

    def test_types(self):
        assert isinstance(self.adain.norm, torch.nn.InstanceNorm2d)
        assert isinstance(self.adain.style_scale, ScaledLinear)
        assert isinstance(self.adain.style_bias, ScaledLinear)

    def test_propagation(self):
        features = torch.randn(10, 256, 4, 4)
        w_noise = torch.randn(10, 512)
        assert self.adain(features, w_noise).shape == (10, 256, 4, 4)
        features = torch.randn(15, 256, 32, 32)
        w_noise = torch.randn(15, 512)
        assert self.adain(features, w_noise).shape == (15, 256, 32, 32)
