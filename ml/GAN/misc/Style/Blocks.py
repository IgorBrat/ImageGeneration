import torch
from ml.GAN.misc.Style.ScaledLayers import WeightedScaledConvo
from ml.GAN.misc.Style.WeightedNoise import InjectWeightedNoise
from ml.GAN.misc.Style.AdaIN import AdaIN


class GeneratorBlock(torch.nn.Module):
    """
    Block with two AdaIN layers and two layers injecting noise for generator, as described in StyleGAN paper
    """

    def __init__(self, in_chan, out_chan, w_chan, device: str = 'cpu'):
        super(GeneratorBlock, self).__init__()
        self.convo1 = WeightedScaledConvo(in_chan, out_chan)
        self.convo2 = WeightedScaledConvo(out_chan, out_chan)
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.noise_inject1 = InjectWeightedNoise(out_chan, device)
        self.noise_inject2 = InjectWeightedNoise(out_chan, device)
        self.adain1 = AdaIN(out_chan, w_chan)
        self.adain2 = AdaIN(out_chan, w_chan)

    def forward(self, feat, w):
        feat = self.adain1(self.lrelu(self.noise_inject1(self.convo1(feat))), w)
        feat = self.adain2(self.lrelu(self.noise_inject2(self.convo2(feat))), w)
        return feat


class Conv2Block(torch.nn.Module):
    """
    Double convolution (down-sample) block for discriminator
    """

    def __init__(self, in_chan, out_chan, leak: float = 0.2):
        super(Conv2Block, self).__init__()
        self.convo1 = WeightedScaledConvo(in_chan, out_chan)
        self.convo2 = WeightedScaledConvo(out_chan, out_chan)
        self.lrelu = torch.nn.LeakyReLU(leak)

    def forward(self, feat):
        return self.lrelu(self.convo2(self.lrelu(self.convo1(feat))))
