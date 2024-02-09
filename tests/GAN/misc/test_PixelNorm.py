import numpy as np
import torch

from ml.GAN.misc.Style.PixelNorm import PixelNormalisation


class TestPixelNormalisation:
    epsilon = 1e-8
    pixel_norm = PixelNormalisation(epsilon)

    def test_attributes(self):
        epsilons = [1e-4, 1e-2, 1e-13]
        for epsilon in epsilons:
            pixel_norm = PixelNormalisation(epsilon)
            assert pixel_norm.eps == epsilon

    def test_propagation(self):
        features = torch.randn(12, 16, 1, 1)
        assert self.pixel_norm(features).shape == (12, 16, 1, 1)
        features = torch.Tensor([[1, 10, -2, -5, 3], [3, 4, 1, 0, 0]])
        assert np.allclose(self.pixel_norm(features),
                           features / torch.sqrt(torch.mean(features ** 2, dim=1, keepdim=True) + self.epsilon))
