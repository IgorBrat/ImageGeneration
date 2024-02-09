import torch
from ml.GAN.misc.Style.ScaledLayers import ScaledLinear
from ml.GAN.misc.Style.PixelNorm import PixelNormalisation


class MultiLayerPerceptron(torch.nn.Module):
    """
    MultiLayer Perceptron used to map input noise to latent space
    """

    def __init__(self, z_chan, w_chan, leak: float = 0.1):
        super(MultiLayerPerceptron, self).__init__()
        self.mlp = torch.nn.Sequential(
            PixelNormalisation(),
            ScaledLinear(z_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
            torch.nn.LeakyReLU(leak),

            ScaledLinear(w_chan, w_chan),
        )

    def forward(self, z):
        return self.mlp(z)
