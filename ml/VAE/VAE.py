import collections
import json
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ml.utils.file_management import check_savefile_integrity
from ml.utils.tensor_logic import reparameterise, get_normal_noise, get_uniform_noise
from ml.utils.data_management import transform_norm
from ml.utils.losses import get_kld_loss
from ml.utils.visual import show_images_norm
from ml.utils.network_misc import init_weights

torch.manual_seed(0)


def get_vae_loss(output, target, mean, log_variance, dataset_shape):
    """
    Calculate loss of variational auto-encoder
    :param output: fake images
    :param target: real images
    :param mean: mean of given features
    :param log_variance: logarithm of standard deviation of given features
    :param dataset_shape: shape of loaded dataset
    :return: VAE loss
    """
    bce_loss = torch.nn.functional.binary_cross_entropy(output,
                                                        target.view(-1, dataset_shape[0] * dataset_shape[1] ** 2))
    kld_loss = get_kld_loss(mean, log_variance)
    return bce_loss + kld_loss


def display_loss(vae_mean_losses, epoch):
    print(f'Epoch {epoch}: VAE loss: {round(vae_mean_losses[-1], 3)}')

    plt.plot(
        range(1, epoch + 1),
        torch.Tensor(vae_mean_losses),
        label="VAE Loss"
    )
    plt.show()


class VaeNet(torch.nn.Module):
    def __init__(self, image_size: int, image_channels: int):
        super(VaeNet, self).__init__()
        self.size = image_size
        self.channels = image_channels

        self.encoder_input = torch.nn.Linear(image_channels * image_size ** 2, 256)
        self.batch_norm_input = torch.nn.BatchNorm1d(256)
        self.encoder_mean = torch.nn.Linear(256, image_size)
        self.encoder_log_variance = torch.nn.Linear(256, image_size)

        # Somehow using dict storing encoder layers gives better visual results from the start,
        # but losses are very high (normally 0-1, there 35)

        # It also doesn't see encoder layers in the model struct, so I guess with dict it does not back-propagate
        # self.encoder = {
        #     ...
        # }
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(image_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(256, image_channels * image_size ** 2),
            torch.nn.Sigmoid(),
        )

    def encode(self, input_images):
        relu_layer = torch.nn.functional.leaky_relu(self.encoder_input(input_images), 0.1)
        return self.encoder_mean(relu_layer), self.encoder_log_variance(relu_layer)

    def decode(self, noise):
        return self.decoder(noise)

    def forward(self, input_images):
        mean, log_variance = self.encode(input_images.view(-1, self.channels * self.size ** 2))
        latent_space_noise = reparameterise(mean, log_variance)
        return self.decode(latent_space_noise), mean, log_variance


class VAE:
    def __init__(self, image_size: int = 64, image_channels: int = 3, presaved: bool = False,
                 save_dir: str = None, params_savefile: str = None, device: str = 'cpu'):
        self.default_savedir = './'
        self.default_savefile = 'vae.pt'
        self.default_params_savefile = 'params.json'
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            image_channels = params['image_channels']
            image_size = params['image_size']
        self.vae_net = VaeNet(image_size, image_channels).to(device)
        self.optim = None
        self.trained_epochs = 0

        self.device = device
        self.init_weights()

    def init_weights(self):
        self.vae_net.apply(init_weights)

    def generate(self, num: int, distribution: str = 'normal'):
        if distribution == 'normal':
            noise = get_normal_noise((num, self.vae_net.channels, self.vae_net.size, self.vae_net.size),
                                     self.device)
        elif distribution == 'uniform':
            noise = get_uniform_noise((num, self.vae_net.channels, self.vae_net.size, self.vae_net.size),
                                      self.device)
        else:
            raise ValueError('Only "normal" and "uniform" distributions are supported for noise')
        images, *_ = self.vae_net(noise)
        return images.view(num, self.vae_net.channels, self.vae_net.size, self.vae_net.size)

    def save(self, save_dir: str = None, savefile: str = None, params_savefile: str = None):
        if not check_savefile_integrity(savefile):
            raise ValueError('File must have one of extensions: .pt, .pth')
        save_dir = save_dir if save_dir else self.default_savedir
        savefile = savefile if savefile else self.default_savefile
        params_savefile = params_savefile if params_savefile else self.default_params_savefile

        torch.save(self.vae_net.state_dict(), os.path.join(save_dir, savefile))
        print(f'Vae saved in "{os.path.join(save_dir, savefile)}"')

        params = {
            'image_channels': self.vae_net.channels,
            'image_size': self.vae_net.size,
        }
        self._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None, savefile: str = None, params_savefile: str = None):
        if not check_savefile_integrity(savefile):
            raise ValueError('File must have one of extensions: .pt, .pth')
        save_dir = save_dir if save_dir else self.default_savedir
        savefile = savefile if savefile else self.default_savefile
        params_savefile = params_savefile if params_savefile else self.default_params_savefile

        torch.save({
            'epoch': self.trained_epochs,
            'model_state_dict': self.vae_net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, os.path.join(save_dir, savefile))
        print(f'Vae saved in "{os.path.join(save_dir, savefile)}"')

        params = {
            'image_channels': self.vae_net.channels,
            'image_size': self.vae_net.size,
        }
        self._save_params(save_dir, params_savefile, params)

    @staticmethod
    def _save_params(save_dir: str, savefile: str, params: dict):
        """
        Save init parameters to specified path
        :param save_dir: save directory
        :param savefile: save file for init params
        :param params: init params dictionary to be saved
        """
        with open(os.path.join(os.getcwd(), save_dir, savefile), 'w') as f_json:
            json.dump(params, f_json)
        print(f'Params saved in {os.path.join(os.getcwd(), save_dir, savefile)}')

    def load(self, save_dir: str = None, savefile: str = None):
        """
        Load model state from specified directory and file
        :param save_dir: save directory
        :param savefile: save file with vae state
        """
        save_dir = save_dir if save_dir else self.default_savedir
        savefile = savefile if savefile else self.default_savefile
        trained_epochs = 0
        if not (check_savefile_integrity(savefile)):
            raise ValueError('File must have one of extensions: .pt, .pth')
        checkpoint = torch.load(os.path.join(save_dir, savefile))
        if isinstance(checkpoint, collections.OrderedDict):
            checkpoint = {'model_state_dict': checkpoint}
        elif isinstance(checkpoint, dict):
            checkpoint = checkpoint
        else:
            raise TypeError('Saved checkpoint should be either "dict" or "OrderedDict"')
        if 'epoch' in checkpoint:
            trained_epochs = checkpoint['epoch']
        self.trained_epochs = trained_epochs

        self.vae_net.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            if not self.optim:
                self.optim = torch.optim.Adam(self.vae_net.parameters())
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Vae loaded successfully')
        print(f'Setting start epoch: {self.trained_epochs}')

    def load_params(self, save_dir: str = None, savefile: str = None):
        """
        Load init params on model creating if specified
        :param save_dir: save directory
        :param savefile: save file with init params
        """
        save_dir = save_dir if save_dir else self.default_savefile
        savefile = savefile if savefile else self.default_params_savefile
        path = os.path.join(os.getcwd(), save_dir, savefile)
        try:
            with open(path) as f_json:
                data = json.load(f_json)
        except FileNotFoundError:
            raise FileNotFoundError(f'Params file at {path} does not exist')
        return data

    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool, show_losses: bool,
              lrn_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self._prepare_for_training(lrn_rate, beta1, beta2)

        vae_losses = []
        vae_mean_losses = []

        self.vae_net.train()

        if show_progress:
            dataloader = tqdm(dataloader)

        start_time = time.time()
        start_epoch = self.trained_epochs if self.trained_epochs else 1
        for epoch in range(start_epoch, start_epoch + epochs):
            for real_images, labels in dataloader:
                real_images = real_images.to(self.device)
                curr_batch = len(real_images)

                self.optim.zero_grad()
                vae_images, mean, log_variance = self.vae_net(real_images)
                vae_loss = get_vae_loss(vae_images, real_images, mean, log_variance,
                                        (self.vae_net.channels, self.vae_net.size))
                vae_loss.backward(retain_graph=True)
                self.optim.step()
                vae_losses.append(vae_loss.item())
            vae_mean_losses.append(np.mean(vae_losses))
            vae_losses.clear()
            if show_losses and num_epochs_to_show and (not epoch % num_epochs_to_show):
                vae_images = vae_images.view(curr_batch, self.vae_net.channels, self.vae_net.size, self.vae_net.size)
                vae_images_norm = transform_norm(vae_images)
                real_images_norm = transform_norm(real_images)
                show_images_norm(vae_images_norm, num=9)
                show_images_norm(real_images_norm, num=9)
                display_loss(vae_mean_losses, epoch)
        duration = (time.time() - start_time) / 60
        print(f'Finished training after {epochs} epochs and', end=' ')
        if duration > 60:
            print(f'{round(duration / 60, 2)} h')
        else:
            print(f'{round(duration, 2)} m')
        print(f'Vae loss: {round(np.mean(vae_mean_losses), 4)}')
        self.vae_net.eval()
        self.trained_epochs += epochs
        vae_mean_losses.clear()

    def _prepare_for_training(self, lrn_rate: float, beta1: float, beta2: float):
        """
        Util method for preparing model and optimiser for training
        :param lrn_rate: optimiser learning rate
        :param beta1, beta2: params of the same name for Adam optimiser
        """
        self.vae_net.train()

        if not self.optim:
            self.optim = torch.optim.Adam(self.vae_net.parameters(), lr=lrn_rate, betas=(beta1, beta2))
        print('Prepared for training VAE')
