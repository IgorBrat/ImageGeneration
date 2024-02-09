import torch


def get_kld_loss(mean, log_variance, Lambda: int = 1):
    """
    Calculate Kullback-Leibler Divergence loss
    :param mean: distribution mean
    :param log_variance: log of distribution standard deviation
    :param Lambda: penalty constant
    :return: KLD loss
    """
    return Lambda * (-0.5) * torch.sum(1 + log_variance - mean ** 2 - log_variance.exp())
