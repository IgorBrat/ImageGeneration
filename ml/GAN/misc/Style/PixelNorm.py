import torch


class PixelNormalisation(torch.nn.Module):
    """
    Pixel normalisation applied to batch of images
    """

    def __init__(self, epsilon: float = 1e-8):
        super(PixelNormalisation, self).__init__()
        self.eps = epsilon

    def forward(self, feat):
        return feat / torch.sqrt(torch.mean(feat ** 2, dim=1, keepdim=True) + self.eps)
