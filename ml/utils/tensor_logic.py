import torch
import scipy


def get_uniform_noise(noise_shape: tuple, device: str):
    """
    Create uniformly distributed random noise vector
    :param noise_shape: shape of resulting vector
    :param device: device to move vector to
    :return: resulting vector
    """
    return torch.rand(*noise_shape, device=device)


def get_normal_noise(noise_shape: tuple, device: str):
    """
    Create normally distributed random noise vector
    :param noise_shape: shape of resulting vector
    :param device: device to move vector to
    :return: resulting vector
    """
    return torch.randn(*noise_shape, device=device)


def get_one_hot_labels(labels: torch.Tensor, num_classes: int):
    """
    Transform given labels to one-hot labels
    :param labels: given labels
    :param num_classes: number of total classes
    :return: one-hot labels
    """
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)


def reparameterise(mean, log_variance):
    """
    Re-parameterise text embedding vector in conditional augmentation
    :param mean: distribution mean
    :param log_variance: log of distribution standard deviation
    :return: re-parameterisation value
    """
    epsilon = torch.exp(0.5 * log_variance)
    return mean + epsilon * log_variance


def matrix_sqrt(mat):
    """
    Calculate matrix square
    :param mat: give matrix
    :return: square of a matrix
    """
    mat_res = mat.cpu().detach().numpy()
    mat_res = scipy.linalg.sqrtm(mat_res)
    return torch.Tensor(mat_res.real, device=mat.device)


def concat_vectors(x, y, dim: int = 1):
    """
    Concat two float vectors
    :param x: first vector
    :param y: second vector
    :param dim: dimension to concat
    :return: concatenated vector
    """
    return torch.cat((x.float(), y.float()), dim=dim)


def gram_matrix(matrix: torch.Tensor):
    """
    Get gram matrix needed for calculating style losses in NST
    :param matrix: matrix to calculate gram matrix of
    :return: gram matrix
    """
    if not matrix.size(dim=0):
        raise ValueError('Parse non-empty matrix')
    if len(matrix.size()) != 2:
        raise TypeError('Matrix must be 2-dimensional')
    return torch.mm(matrix, matrix.t())
