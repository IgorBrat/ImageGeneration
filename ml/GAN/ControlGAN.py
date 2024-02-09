from typing import List

import torch
import time
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ml.utils.network_misc import ResBlock, init_weights
from ml.utils.tensor_logic import get_uniform_noise, get_one_hot_labels, concat_vectors, get_normal_noise
from ml.utils.file_management import check_savefile_integrity
from ml.GAN.GAN import GAN


# Authors of the paper stick with the uniform distribution (-1,1)

class Generator(torch.nn.Module):
    def __init__(self, input_channels: int, out_channels: int = 3) -> None:
        super(Generator, self).__init__()
        self.input_channels = input_channels
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_channels, 64, kernel_size=1),
            ResBlock(64),
            ResBlock(64),
            self.deconv_block(64, 64, kernel_size=5, stride=2, padding=2),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.deconv_block(64, 64, kernel_size=5, stride=2, padding=2),
            ResBlock(64),
            ResBlock(64),
            self.deconv_block(64, out_channels, kernel_size=5, padding=2, is_last=True),
        )

    @staticmethod
    def deconv_block(in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 1,
                     padding: int = 0, leak: float = 0.2, is_last: bool = False):
        if not is_last:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(leak),
            )
        else:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.Tanh(),
            )

    def forward(self, noise):
        return self.model(noise)


class Discriminator(torch.nn.Module):
    def __init__(self, image_channels: int = 3) -> None:
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 64, kernel_size=5, stride=2, padding=2),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 128),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        )

    @staticmethod
    def pooling_block(kernel_size: int = 4, stride: int = 1,
                      padding: int = 0, leak: float = 0.1):
        return torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size, stride, padding),
            torch.nn.LeakyReLU(leak),
        )

    def forward(self, noise):
        return self.model(noise)


class Classifier(torch.nn.Module):
    def __init__(self, num_features: int, image_channels: int = 3) -> None:
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 64, kernel_size=5, stride=2, padding=2),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 128),
            torch.nn.Linear(128, num_features),
            torch.nn.Sigmoid(),
        )

    @staticmethod
    def pooling_block(kernel_size: int = 4, stride: int = 1,
                      padding: int = 0, leak: float = 0.1):
        return torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size, stride, padding),
            torch.nn.LeakyReLU(leak),
        )

    def forward(self, noise):
        return self.model(noise)


class ControlGAN(GAN):
    def __init__(self, noise_channels: int = 0, features: int = 0, image_channels: int = 3, presaved: bool = False,
                 save_dir: str = None, params_savefile: str = None, device: str = 'cpu'):
        if not presaved and not features:
            raise ValueError('Model must be either pre-saved or have positive number of features')
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            image_channels = params['image_channels']
            features = params['num_features']
        super().__init__(noise_channels, image_channels, presaved, device=device)
        self.features = features
        self.gen = Generator(noise_channels + features, image_channels)
        self.disc = Discriminator(image_channels)
        self.classif = Classifier(features, image_channels)
        self.classif_optim = None
        # To get 3x128x128 images
        self.noise_size = 32
        self.default_classif_savefile = 'classif.pt'
        self.trained_epochs_classif = 0

        self.init_models()

    def init_models(self):
        super().init_models()
        try:
            self.classif.apply(init_weights)
        except Exception as e:
            raise RuntimeError(f'Can`t init classifier weights.\nException: {e}')
        finally:
            print('Initialised classifier weights')

    def generate(self, num: int, feature: int, distribution: str = 'normal'):
        """
        Generate images with given features
        :param num: number of generated images
        :param feature: desired feature
        :param distribution: input noise distribution, one of ['normal', 'uniform']
        :return: generated images
        """
        if distribution == 'normal':
            noise = get_normal_noise((num, self.noise_channels, self.noise_size, self.noise_size), self.device)
        elif distribution == 'uniform':
            noise = get_uniform_noise((num, self.noise_channels, self.noise_size, self.noise_size), self.device)
        else:
            raise ValueError('Only "normal" and "uniform" distributions are supported for noise')

        feature = torch.Tensor([feature]).long()
        feature = get_one_hot_labels(feature, self.features)
        feature = feature[:, :, None, None]
        feature = feature.repeat(num, 1, self.noise_size, self.noise_size)

        noise_and_features = concat_vectors(noise, feature)
        return self.gen(noise_and_features)

    def save(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
             classif_savefile: str = None, params_savefile: str = None):
        """
        Save model parameters to specified directory and files
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param classif_savefile: save file for classifier
        :param params_savefile: save file for init params
        """
        if not check_savefile_integrity(classif_savefile):
            raise ValueError('File must have one of extensions: .pt, .pth')
        super().save(save_dir, gen_savefile, disc_savefile, save_params=False)
        save_dir = save_dir if save_dir else self.default_savedir
        classif_savefile = classif_savefile if classif_savefile else self.default_classif_savefile

        torch.save(self.classif.state_dict(), os.path.join(save_dir, classif_savefile))
        print(f'Classifier saved in "{os.path.join(save_dir, classif_savefile)}"')

        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'num_features': self.features,
        }
        self._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
                          classif_savefile: str = None, params_savefile: str = None):
        """
        Save model parameters to specified directory and files with training history
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param classif_savefile: save file for classifier
        :param params_savefile: save file for init params
        """
        if not check_savefile_integrity(classif_savefile):
            raise ValueError('File must have one of extensions: .pt, .pth')
        super().save_with_history(save_dir, gen_savefile, disc_savefile, save_params=False)
        if not self.trained_epochs_classif:
            raise ValueError('Classifier was not trained, so can not be saved with history')
        save_dir = save_dir if save_dir else self.default_savedir
        classif_savefile = classif_savefile if classif_savefile else self.default_classif_savefile
        torch.save({
            'epoch': self.trained_epochs_classif,
            'model_state_dict': self.classif.state_dict(),
            'optimizer_state_dict': self.classif_optim.state_dict(),
        }, os.path.join(save_dir, classif_savefile))
        print(f'Classifier history saved in "{os.path.join(save_dir, classif_savefile)}"')

        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'num_features': self.features,
        }
        self._save_params(save_dir, params_savefile, params)

    def load(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
             classif_savefile: str = None):
        """
        Load model state from specified directory and files
        :param save_dir: save directory
        :param gen_savefile: save file with generator state
        :param disc_savefile: save file with discriminator state
        :param classif_savefile: save file with classifier state
        """
        super().load(save_dir, gen_savefile, disc_savefile)
        save_dir = save_dir if save_dir else self.default_savedir
        classif_savefile = classif_savefile if classif_savefile else self.default_classif_savefile
        classif_checkpoint = self._load(save_dir, classif_savefile)
        if 'epoch' in classif_checkpoint:
            self.trained_epochs_classif = classif_checkpoint['epoch']
        self.classif.load_state_dict(classif_checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in classif_checkpoint:
            if not self.classif_optim:
                self.classif_optim = torch.optim.Adam(self.gen.parameters())
            self.classif_optim.load_state_dict(classif_checkpoint['optimizer_state_dict'])
        print('Classifier loaded successfully')
        print(f'Setting classifier start epoch: {self.trained_epochs_classif}')

    def train_classifier(self, dataloader, epochs, num_epochs_to_show, show_progress: bool = True,
                         show_losses: bool = True, lrn_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        """
        Train classifier independently of generator and discriminator
        :param dataloader: Dataloader object containing dataset
        :param epochs: number of epochs to train model for
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :param show_losses: bool value indicating if losses have to be shown
        :param lrn_rate: optimiser learning rate
        :param beta1: param of the same name for Adam optimiser
        :param beta2: param of the same name for Adam optimiser
        """
        self.classif.train()

        loss_crit = torch.nn.BCELoss()
        if not self.classif_optim:
            self.classif_optim = torch.optim.Adam(self.gen.parameters(), lr=lrn_rate, betas=(beta1, beta2))

        if show_progress:
            dataloader = tqdm(dataloader)

        classif_losses = []
        classif_mean_losses = []

        start_time = time.time()
        start_epoch = self.trained_epochs_classif + 1 if self.trained_epochs_classif else 1
        for epoch in range(start_epoch, start_epoch + epochs):
            for real_images, *rest in dataloader:
                # TODO: Check curr-batch ???
                feature_labels = rest[0]
                real_images = real_images.to(self.device)
                feature_labels = feature_labels.to(self.device)
                feature_labels = get_one_hot_labels(feature_labels, num_classes=self.features).float()

                self.classif_optim.zero_grad()
                classif_out = self.classif(real_images)
                assert classif_out.shape == feature_labels.shape

                classif_loss = loss_crit(classif_out, feature_labels)
                classif_loss.backward()
                self.classif_optim.step()
                classif_losses.append(classif_loss.item())
            classif_mean_losses.append(sum(classif_losses) / len(classif_losses))
            classif_losses.clear()
            if show_losses and num_epochs_to_show and (not epoch % num_epochs_to_show):
                print(f'Classifier loss: {classif_mean_losses[-1]}')
                plt.plot(
                    range(start_epoch, start_epoch + epoch),
                    torch.Tensor(classif_mean_losses),
                    label="Classifier Loss",
                )
                plt.show()
        duration = (time.time() - start_time) / 60
        print(f'Finished training after {epochs} epochs and', end=' ')
        if duration > 60:
            print(f'{round(duration / 60, 2)} h')
        else:
            print(f'{round(duration, 2)} m')
        print(f"Mean classifier loss: {sum(classif_mean_losses) / len(classif_mean_losses)}")
        self.classif.eval()
        classif_mean_losses.clear()

    def get_disc_loss(self, criterion, real, fake, alpha):
        """
        Calculate discriminator loss
        :param criterion: criterion object to calculate losses
        :param real: real images
        :param fake: fake (generated) images
        :param alpha: controls proportion of how crucial for us real/fake losses
        :return: discriminator loss
        """
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha parameter for discriminator loss must be in range [0, 1]')
        real_loss = criterion(real, torch.ones_like(real))
        fake_loss = criterion(fake, torch.zeros_like(fake))
        return alpha * real_loss + (1 - alpha) * fake_loss

    def get_gen_loss(self, criterion, classifier_output, disc_output, feature_labels,
                     gamma):
        """
        Calculate generator loss
        :param criterion: criterion object to calculate losses
        :param classifier_output: classifier prediction
        :param disc_output: discriminator prediction
        :param feature_labels: one-hot labels of present features
        :param gamma: controls how much attention we want to give to features
        :return: generator loss
        """
        classifier_loss = criterion(classifier_output, feature_labels)
        disc_loss = criterion(disc_output, torch.ones_like(disc_output))
        return gamma * classifier_loss + disc_loss, classifier_loss

    def train(self, dataloader, epochs, train_classif: bool = True, classif_epochs: int = 10,
              num_epochs_to_show: int = None, show_progress: bool = True, show_losses: bool = True,
              lrn_rate1: float = 1e-4, lrn_rate2: float = 2e-5, beta1: float = 0.9, beta2: float = 0.999,
              alpha: float = 0.5, gamma: float = 0, gamma_lr_rate: float = 1e-3, equilibrium: float = 0.05):
        """
        Train method
        :param dataloader: Dataloader object containing dataset
        :param epochs: number of epochs to train model for
        :param train_classif:
        :param classif_epochs:
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :param show_losses: bool value indicating if losses have to be shown
        :param lrn_rate1: classifier optimiser learning rate
        :param lrn_rate2: generator/discriminator optimiser learning rate
        :param beta1: param of the same name for Adam optimiser
        :param beta2: param of the same name for Adam optimiser
        :param alpha: controls proportion of how crucial for us real/fake losses
        :param gamma: controls how much attention we want to give to features
        :param gamma_lr_rate: gamma learning rate
        :param equilibrium: equilibrium controlling learning of gamma
        """
        if train_classif and classif_epochs:
            self.train_classifier(dataloader, classif_epochs, int(classif_epochs ** 0.5), show_progress, show_losses,
                                  lrn_rate2, beta1, beta2)
        else:
            print('Using pre-trained classifier')
        self._prepare_for_training(lrn_rate1, beta1, beta2)
        disc_losses = []
        gen_losses = []
        disc_mean_losses = []
        gen_mean_losses = []

        loss_crit = torch.nn.BCELoss()
        if show_progress:
            dataloader = tqdm(dataloader)

        start_time = time.time()
        start_epoch = self.trained_epochs_classif if self.trained_epochs_classif else 1
        for epoch in range(start_epoch, epochs + start_epoch + 1):
            for real_images, *rest in dataloader:
                feature_labels = rest[0]
                curr_batch = len(real_images)
                real_images = real_images.to(self.device)
                feature_labels = feature_labels.to(self.device)
                print(feature_labels)
                feature_labels = get_one_hot_labels(feature_labels, num_classes=self.features).float()

                # Fill each channel with 0-s or 1-s depending on existing feature label
                # Meaning 128x128x40 labels matrix per image (40 features, 128x128 filled 0-s or 1-s)
                feature_labels_expanded = feature_labels[:, :, None, None]
                feature_labels_expanded = feature_labels_expanded.repeat(1, 1, self.noise_size, self.noise_size)
                assert feature_labels_expanded.shape == (
                    curr_batch, self.features, self.noise_size, self.noise_size)
                print(feature_labels_expanded)

                # DISCRIMINATOR
                self.disc_optim.zero_grad()
                noise = get_uniform_noise((curr_batch, self.noise_channels, self.noise_size, self.noise_size),
                                          self.device)
                noise_with_features = concat_vectors(noise, feature_labels_expanded)
                assert noise_with_features.shape == (
                    curr_batch, self.features + self.noise_channels, self.noise_size, self.noise_size)
                fake_images = self.gen(noise_with_features)
                assert fake_images.shape == real_images.shape

                fake_pred = self.disc(fake_images)
                real_pred = self.disc(real_images)

                disc_loss = self.get_disc_loss(loss_crit, real_pred, fake_pred, alpha)
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()
                disc_losses.append(disc_loss.item())

                # GENERATOR
                self.gen_optim.zero_grad()
                disc_pred = self.disc(fake_images)
                classif_fake_pred = self.classif(fake_images)
                classif_real_pred = self.classif(real_images)

                gen_loss, classif_fake_loss = self.get_gen_loss(loss_crit, classif_fake_pred, disc_pred, feature_labels,
                                                                gamma)
                classif_real_loss = loss_crit(classif_real_pred, feature_labels)
                # Update gamma 
                gamma += gamma_lr_rate * (classif_fake_loss.item() - equilibrium * classif_real_loss.item())
                gen_loss.backward()
                self.gen_optim.step()
                gen_losses.append(gen_loss.item())
            self._compute_results_per_epoch(epoch, num_epochs_to_show, fake_images, real_images, gen_losses,
                                            disc_losses, gen_mean_losses, disc_mean_losses, show_losses)
        self._display_results_of_training(epochs, start_time, gen_mean_losses, disc_mean_losses)
