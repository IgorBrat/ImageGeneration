import torch


class ScaledLinear(torch.nn.Module):
    """
    Weighted Scaled Linear layer used in Mapping Network
    """

    def __init__(self, in_feat, out_feat):
        super(ScaledLinear, self).__init__()
        self.layer = torch.nn.Linear(in_feat, out_feat)
        self.scale = (2 / in_feat) ** 0.5
        self.unscaled_bias = self.layer.bias
        self.layer.bias = None

        torch.nn.init.normal_(self.layer.weight)
        torch.nn.init.zeros_(self.unscaled_bias)

    def forward(self, feat):
        return self.layer(feat * self.scale) + self.unscaled_bias


class WeightedScaledConvo(torch.nn.Module):
    """
    Weighted Scaled Convolutional layer used to equalise learning rate
    """

    def __init__(self, in_feat, out_feat, kernel: int = 3, stride: int = 1, padding: int = 1):
        super(WeightedScaledConvo, self).__init__()
        self.layer = torch.nn.Conv2d(in_feat, out_feat, kernel, stride, padding)
        self.scale = (2 / (in_feat * kernel ** 2)) ** 0.5
        self.unscaled_bias = self.layer.bias
        self.layer.bias = None

        torch.nn.init.normal_(self.layer.weight)
        torch.nn.init.zeros_(self.unscaled_bias)

    def forward(self, feat):
        return self.layer(feat * self.scale) + self.unscaled_bias.view(1, self.unscaled_bias.shape[0], 1, 1)
