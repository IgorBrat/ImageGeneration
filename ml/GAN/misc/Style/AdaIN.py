import torch
from ml.GAN.misc.Style.ScaledLayers import ScaledLinear


class AdaIN(torch.nn.Module):
    """
    Adaptive Instance Normalisation layer described in StyleGAN research paper.
    Used to propagate latent w-space noise to generator
    """

    def __init__(self, feat_chan, w_chan):
        super(AdaIN, self).__init__()
        self.norm = torch.nn.InstanceNorm2d(feat_chan)
        self.style_scale = ScaledLinear(w_chan, feat_chan)
        self.style_bias = ScaledLinear(w_chan, feat_chan)

    def forward(self, feat, w):
        """
        Normalise each feature map separately
        :param feat: features from input z-space noise
        :param w: latent w-space noise
        :return: normalised (per each instance) vector
        """
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * self.norm(feat) + style_bias
