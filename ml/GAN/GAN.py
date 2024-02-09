import collections
import json
from abc import ABC, abstractmethod
import torch
import os
import time

from ml.utils.file_management import check_savefile_integrity
from ml.utils.network_misc import init_weights, count_parameters
from ml.utils.tensor_logic import get_uniform_noise, get_normal_noise
from ml.utils.visual import display_losses, show_images_norm


class GAN(ABC):
    """
    Generative Adversarial Network abstract class
    """

    def __init__(self, noise_channels, image_channels, presaved, device):
        """
        :param noise_channels: channels of input noise vector
        :param image_channels: channels of output images (3 for RGB)
        :param presaved: bool value indicating if model is saved locally
        :param device: device to move model to
        """
        if not presaved and not noise_channels:
            raise ValueError('Model must be either pre-saved or have positive number of noise channels')
        self.noise_channels = noise_channels
        self.image_channels = image_channels
        self.gen = None
        self.disc = None
        self.trained_epochs = 0
        self.gen_optim, self.disc_optim = None, None
        self.device = device

        self.default_savedir = './'
        self.default_gen_savefile = 'gen.pt'
        self.default_disc_savefile = 'disc.pt'
        self.default_params_savefile = 'params.json'

    def init_models(self):
        """
        Initialise model's weights to speed up training
        """
        try:
            self.gen.apply(init_weights)
            self.disc.apply(init_weights)
        except Exception as e:
            raise RuntimeError(f'Can`t init weights.\nException: {e}')
        finally:
            print('Initialised model weights')

    def eval(self):
        """
        Set model to evaluation mode
        """
        self.gen.eval()
        self.disc.eval()

    def generate(self, num: int, distribution: str = 'normal'):
        """
        Generate images
        :param num: number of generated images
        :param distribution: input noise distribution, one of ['normal', 'uniform']
        :return: generated images
        """
        if distribution == 'normal':
            noise = get_normal_noise((num, self.noise_channels), self.device)
        elif distribution == 'uniform':
            noise = get_uniform_noise((num, self.noise_channels), self.device)
        else:
            raise ValueError('Only "normal" and "uniform" distributions are supported for noise')
        return self.gen(noise)

    def save(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
             save_params: bool = True, params_savefile: str = None):
        """
        Save model parameters to specified directory and files
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param save_params: bool value indicating if init params must be saved
        :param params_savefile: save file for init params
        """
        if not (check_savefile_integrity(gen_savefile) and check_savefile_integrity(disc_savefile)):
            raise ValueError('File must have one of extensions: .pt, .pth')
        save_dir = save_dir if save_dir else self.default_savedir
        gen_savefile = gen_savefile if gen_savefile else self.default_gen_savefile
        disc_savefile = disc_savefile if disc_savefile else self.default_disc_savefile
        params_savefile = params_savefile if params_savefile else self.default_params_savefile

        torch.save(self.gen.state_dict(), os.path.join(save_dir, gen_savefile))
        print(f'Generator saved in "{os.path.join(save_dir, gen_savefile)}"')
        torch.save(self.disc.state_dict(), os.path.join(save_dir, disc_savefile))
        print(f'Discriminator saved in "{os.path.join(save_dir, disc_savefile)}"')

        if save_params:
            params = {
                'noise_channels': self.noise_channels,
                'image_channels': self.image_channels,
            }
            self._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
                          save_params: bool = True, params_savefile: str = None):
        """
        Save model parameters to specified directory and files with training history
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param save_params: bool value indicating if init params must be saved
        :param params_savefile: save file for init params
        """
        if not (check_savefile_integrity(gen_savefile) and check_savefile_integrity(disc_savefile)):
            raise ValueError('File must have one of extensions: .pt, .pth')
        if not self.trained_epochs:
            raise ValueError('Generator/discriminator were not trained, so can not be saved with history')
        save_dir = save_dir if save_dir else self.default_savedir
        gen_savefile = gen_savefile if gen_savefile else self.default_gen_savefile
        disc_savefile = disc_savefile if disc_savefile else self.default_disc_savefile
        params_savefile = params_savefile if params_savefile else self.default_params_savefile

        torch.save({
            'epoch': self.trained_epochs,
            'model_state_dict': self.gen.state_dict(),
            'optimizer_state_dict': self.gen_optim.state_dict(),
        }, os.path.join(save_dir, gen_savefile))
        print(f'Generator history saved in "{os.path.join(save_dir, gen_savefile)}"')
        torch.save({
            'epoch': self.trained_epochs,
            'model_state_dict': self.disc.state_dict(),
            'optimizer_state_dict': self.disc_optim.state_dict(),
        }, os.path.join(save_dir, disc_savefile))
        print(f'Discriminator history saved in "{os.path.join(save_dir, disc_savefile)}"')

        if save_params:
            params = {
                'noise_channels': self.noise_channels,
                'image_channels': self.image_channels,
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

    def load(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None):
        """
        Load model state from specified directory and files
        :param save_dir: save directory
        :param gen_savefile: save file with generator state
        :param disc_savefile: save file with discriminator state
        """
        save_dir = save_dir if save_dir else self.default_savedir
        gen_savefile = gen_savefile if gen_savefile else self.default_gen_savefile
        disc_savefile = disc_savefile if disc_savefile else self.default_disc_savefile
        trained_epochs_gen = 0
        trained_epochs_disc = 0
        gen_checkpoint = self._load(save_dir, gen_savefile)
        if 'epoch' in gen_checkpoint:
            trained_epochs_gen = gen_checkpoint['epoch']
        disc_checkpoint = self._load(save_dir, disc_savefile)
        if 'epoch' in disc_checkpoint:
            trained_epochs_disc = gen_checkpoint['epoch']
        if trained_epochs_gen != trained_epochs_disc:
            raise AssertionError('Generator and Discriminator can`t be trained for different amount of epochs.')
        self.trained_epochs = trained_epochs_gen

        self.gen.load_state_dict(gen_checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in gen_checkpoint:
            if not self.gen_optim:
                self.gen_optim = torch.optim.Adam(self.gen.parameters())
            self.gen_optim.load_state_dict(gen_checkpoint['optimizer_state_dict'])
        print('Generator loaded successfully')

        self.disc.load_state_dict(disc_checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in disc_checkpoint:
            if not self.disc_optim:
                self.disc_optim = torch.optim.Adam(self.disc.parameters())
            self.disc_optim.load_state_dict(disc_checkpoint['optimizer_state_dict'])
        print('Discriminator loaded successfully')
        print(f'Setting start epoch: {self.trained_epochs}')

    @staticmethod
    def load_params(save_dir: str = None, savefile: str = None):
        """
        Load init params on model creating if specified
        :param save_dir: save directory
        :param savefile: save file with init params
        """
        save_dir = save_dir if save_dir else r'./'
        savefile = savefile if savefile else 'params.json'
        path = os.path.join(os.getcwd(), save_dir, savefile)
        try:
            with open(path) as f_json:
                data = json.load(f_json)
        except FileNotFoundError:
            raise FileNotFoundError(f'Params file at {path} does not exist')
        return data

    def _load(self, save_dir: str, savefile: str):
        """
        Util method for loading network state
        :param save_dir: save directory
        :param savefile: save file with model state
        :return: checkpoint with model state
        """
        if not (check_savefile_integrity(savefile)):
            raise ValueError('File must have one of extensions: .pt, .pth')
        checkpoint = torch.load(os.path.join(save_dir, savefile))
        if isinstance(checkpoint, collections.OrderedDict):
            return {'model_state_dict': checkpoint}
        elif isinstance(checkpoint, dict):
            return checkpoint
        else:
            raise TypeError('Saved checkpoint should be either "dict" or "OrderedDict"')

    @abstractmethod
    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool, show_losses: bool,
              lrn_rate: float, beta1: float, beta2: float):
        """
        Train method
        :param dataloader: Dataloader object containing dataset
        :param epochs: number of epochs to train model for
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :param show_losses: bool value indicating if losses have to be shown
        :param lrn_rate: optimiser learning rate
        :param beta1: param of the same name for Adam optimiser
        :param beta2: param of the same name for Adam optimiser
        """
        pass

    def _prepare_for_training(self, lrn_rate: float, beta1: float, beta2: float):
        """
        Util method for preparing model and optimiser for training
        :param lrn_rate: optimiser learning rate
        :param beta1, beta2: params of the same name for Adam optimiser
        """
        self.gen.train()
        self.disc.train()

        if not self.gen_optim:
            self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lrn_rate, betas=(beta1, beta2))
        if not self.disc_optim:
            self.disc_optim = torch.optim.Adam(self.disc.parameters(), lr=lrn_rate, betas=(beta1, beta2))
        print('Prepared for training generator/discriminator')

    def _display_results_of_training(self, epochs, start_time, gen_mean_losses, disc_mean_losses):
        """
        Util method for displaying results of training
        :param epochs: number of total training epochs
        :param start_time: starting time of training
        :param gen_mean_losses: mean losses of generator
        :param disc_mean_losses: mean losses of discriminator
        """
        duration = (time.time() - start_time) / 60
        print(f'Finished training after {epochs} epochs and', end=' ')
        if duration > 60:
            print(f'{round(duration / 60, 2)} h')
        else:
            print(f'{round(duration, 2)} m')
        print(f'Mean generator loss: {round(sum(gen_mean_losses) / len(gen_mean_losses), 4)}')
        print(f'Mean discriminator loss: {round(sum(disc_mean_losses) / len(disc_mean_losses), 4)}')
        self.eval()
        self.trained_epochs += epochs
        gen_mean_losses.clear()
        disc_mean_losses.clear()

    def _compute_results_per_epoch(self, epoch, num_epochs_to_show, fake_images, real_images, gen_losses, disc_losses,
                                   gen_mean_losses, disc_mean_losses, show_losses):
        """
        Compute intermediate losses and display current results of training
        :param epoch: current epoch
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param fake_images: last generated images batch
        :param real_images: last real images batch
        :param gen_losses: losses of generator
        :param disc_losses: losses of discriminator
        :param gen_mean_losses: list of mean generator losses
        :param disc_mean_losses: list of mean discriminator losses
        :param show_losses: bool value indicating if losses have to be shown
        """
        gen_mean_losses.append(sum(gen_losses) / len(gen_losses))
        disc_mean_losses.append(sum(disc_losses) / len(disc_losses))
        gen_losses.clear()
        disc_losses.clear()
        if epoch == 1:
            print('Model successfully passed 1-st epoch.')
        if show_losses and num_epochs_to_show and (not epoch % num_epochs_to_show):
            show_images_norm(fake_images, num=9)
            show_images_norm(real_images, num=9)
            display_losses(gen_mean_losses, disc_mean_losses, epoch)

    def count_parameters(self):
        """
        Count and display number of models parameters
        """
        gen_train_params, gen_fixed_params = count_parameters(self.gen)
        disc_train_params, disc_fixed_params = count_parameters(self.disc)

        print(f'GEN: Trainable params {gen_train_params}, Fixed params {gen_fixed_params}')
        print(f'DISC: Trainable params {disc_train_params}, Fixed params {disc_fixed_params}')
