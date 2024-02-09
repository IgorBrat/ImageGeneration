import torch
from ml.utils.tensor_logic import get_normal_noise


class InjectWeightedNoise(torch.nn.Module):
    """
    Inject weighted random noise in different layers of StyleGAN generator to control features at each level
    """

    def __init__(self, in_chan, device: str = 'cpu'):
        super(InjectWeightedNoise, self).__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.zeros(1, in_chan, 1, 1))

    def forward(self, feat):
        randn_noise = get_normal_noise((feat.shape[0], 1, feat.shape[2], feat.shape[3]), self.device)
        return feat + self.weight * randn_noise
