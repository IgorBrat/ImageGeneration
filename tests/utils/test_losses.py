import numpy as np
import torch

from ml.utils.losses import get_kld_loss


class TestKLD:
    mean = torch.Tensor([3])
    log_variance = torch.Tensor([5])
    mean2 = torch.Tensor([10])
    log_variance2 = torch.Tensor([0.1])

    def test_zero_lambda(self):
        assert not get_kld_loss(self.mean, self.log_variance, Lambda=0)

    def test_kld(self):
        assert np.isclose(get_kld_loss(self.mean, self.log_variance, Lambda=1), 75.706, atol=1e-3, rtol=0)
        assert np.isclose(get_kld_loss(self.mean2, self.log_variance2, Lambda=2), 100.005, atol=1e-3, rtol=0)
