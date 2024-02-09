from torch import Tensor
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_images_norm(images: Tensor, num: int = 0, num_row: int = 0):
    """
    Display images with normalisation
    :param images: given images
    :param num: number of images to show
    :param num_row: number of images per row
    """
    if not num:
        num = len(images)
    if not num_row:
        num_row = num ** 0.5
    images = (images + 1) / 2
    images = images.detach().cpu()
    image_grid = make_grid(images[:num], nrow=num_row)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def show_images_unnorm(images: Tensor, num: int = 1, num_row: int = 3):
    """
    Display images without normalisation
    :param images: given images
    :param num: number of images to show
    :param num_row: number of images per row
    """
    if not num:
        num = len(images)
    if not num_row:
        num_row = num ** 0.5
    images = images.detach().cpu()
    image_grid = make_grid(images[:num], nrow=num_row)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def display_losses(gen_mean_losses, disc_mean_losses, epoch, start_epoch: int = 1):
    """
    Display current losses of generator and discriminator
    :param gen_mean_losses: mean losses of generator
    :param disc_mean_losses: mean losses of discriminator
    :param epoch: current epoch
    :param start_epoch: starting epoch
    """
    print(
        f"Epoch {epoch}: Generator loss: {round(gen_mean_losses[-1], 3)}, discriminator loss: {round(disc_mean_losses[-1], 3)}")

    plt.plot(
        range(start_epoch, epoch + start_epoch),
        Tensor(gen_mean_losses),
        label="Generator Loss"
    )
    plt.plot(
        range(start_epoch, epoch + start_epoch),
        Tensor(disc_mean_losses),
        label="Discriminator Loss"
    )
    plt.legend()
    plt.show()
