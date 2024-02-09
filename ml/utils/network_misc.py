import torch
import numpy as np


def count_parameters(model):
    """
    Count trainable and static parameters of given model
    :param model: given model (gen/disc/classifier)
    :return: Trainable and static parameters of given model
    """
    model_train_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_static_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    train_params = sum([np.prod(p.size()) for p in model_train_parameters])
    static_params = sum([np.prod(p.size()) for p in model_static_parameters])
    return train_params, static_params


def init_weights(layer):
    """
    Initialise weights of a given layer for training speed up
    :param layer: given layer
    """
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d) or isinstance(layer,
                                                                                                       torch.nn.Linear):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    if isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 1.0, 0.02)
        torch.nn.init.constant_(layer.bias, 0)


class ResBlock(torch.nn.Module):
    """
    Residual block for Controllable and Stack GANs
    """

    def __init__(self, channels, leak: float = 0.1):
        super(ResBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(channels, channels, 3, 1, 1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ConvTranspose2d(channels, channels, 3, 1, 1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.LeakyReLU(leak),
        )

    def size(self):
        return len(self.block)

    def forward(self, noise):
        residual = noise
        out = self.block(noise)
        return out + residual
