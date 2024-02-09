import time
import torch
from tqdm.auto import tqdm
from ml.utils.tensor_logic import get_uniform_noise, get_one_hot_labels, concat_vectors, get_normal_noise
from ml.GAN.GAN import GAN


class Generator(torch.nn.Module):
    def __init__(self, noise_channels: int, out_channels: int = 3) -> None:
        super(Generator, self).__init__()
        self.noise_channels = noise_channels
        self.model = torch.nn.Sequential(
            self.gen_block(noise_channels, 256),
            self.gen_block(256, 128, stride=2, padding=1),
            self.gen_block(128, 64, stride=2, padding=1),
            self.gen_block(64, 32, stride=2, padding=1),
            self.gen_block(32, out_channels, stride=2, padding=1, is_last=True),
        )

    @staticmethod
    def gen_block(in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 1,
                  padding: int = 0, leaky_relu_slope: float = 0.2, is_last: bool = False):
        if not is_last:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(leaky_relu_slope),
            )
        else:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.Tanh(),
            )
        return block

    def forward(self, noise):
        noise = noise.view(len(noise), self.noise_channels, 1, 1)
        return self.model(noise)


class Discriminator(torch.nn.Module):
    def __init__(self, image_channels: int = 3) -> None:
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            self.disc_block(image_channels, 32, stride=2, padding=2),
            self.disc_block(32, 64, stride=2, padding=2),
            self.disc_block(64, 128, stride=2, padding=2),
            self.disc_block(128, 256, stride=2, padding=2),
            self.disc_block(256, 1, kernel_size=4, is_last=True),
        )

    @staticmethod
    def disc_block(in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1,
                   padding: int = 0, leaky_relu_slope: float = 0.2, is_last: bool = False):
        if not is_last:
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(leaky_relu_slope),
            )
        else:
            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.LeakyReLU(leaky_relu_slope),
                torch.nn.Flatten(),
                torch.nn.Sigmoid(),
            )
        return block

    def forward(self, noise):
        return self.model(noise)


class CondGAN(GAN):
    """
    Conditional GAN model utilising classes of generated objects
    """

    def __init__(self, noise_channels: int = 100, num_classes: int = 0, image_channels: int = 3, presaved: bool = False,
                 save_dir: str = None, params_savefile: str = None, device: str = 'cpu'):
        if not presaved and not num_classes:
            raise ValueError('Model must be either pre-saved or have positive number of classes')
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            image_channels = params['image_channels']
            num_classes = params['num_classes']
        super().__init__(noise_channels=noise_channels, image_channels=image_channels, presaved=False, device=device)
        self.gen = Generator(noise_channels + num_classes, image_channels).to(device)
        self.disc = Discriminator(image_channels + num_classes).to(device)
        self.image_size = 64
        self.classes = num_classes

        self.init_models()

    def generate(self, num: int, target_class: int, distribution: str = 'normal'):
        """
        Generate images of given class
        :param num: number of generated images
        :param target_class: desired class
        :param distribution: input noise distribution, one of ['normal', 'uniform']
        :return: generated images
        """
        label = get_one_hot_labels(torch.Tensor([target_class]).long(), self.classes)
        print(label.shape)
        label = label.repeat(num, 1)
        print(label.shape)
        if distribution == 'normal':
            noise = get_normal_noise((num, self.noise_channels), self.device)
        elif distribution == 'uniform':
            noise = get_uniform_noise((num, self.noise_channels), self.device)
        else:
            raise ValueError('Only "normal" and "uniform" distributions are supported for noise')
        noise_and_labels = concat_vectors(noise, label)
        return self.gen(noise_and_labels)

    def save(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
             params_savefile: str = None):
        """
        Save model parameters to specified directory and files
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param params_savefile: save file for init params
        """
        super().save(save_dir, gen_savefile, disc_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        print(params_savefile)
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'num_classes': self.classes,
        }
        self._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
                          params_savefile: str = None):
        """
        Save model parameters to specified directory and files with training history
        :param save_dir: save directory
        :param gen_savefile: save file for generator
        :param disc_savefile: save file for discriminator
        :param params_savefile: save file for init params
        """
        super().save_with_history(save_dir, gen_savefile, disc_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'num_classes': self.classes,
        }
        self._save_params(save_dir, params_savefile, params)

    def train(self, dataloader, epochs, num_epochs_to_show: int = None, show_progress: bool = True,
              show_losses: bool = True, lrn_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self._prepare_for_training(lrn_rate, beta1, beta2)

        disc_losses = []
        gen_losses = []
        disc_mean_losses = []
        gen_mean_losses = []

        loss_crit = torch.nn.BCELoss()
        if show_progress:
            dataloader = tqdm(dataloader)

        start_time = time.time()
        start_epoch = self.trained_epochs if self.trained_epochs else 1
        for epoch in range(start_epoch, start_epoch + epochs):
            for real_images, *rest in dataloader:
                labels = rest[0]
                curr_batch = len(real_images)
                real_images = real_images.to(self.device)

                one_hot_labels = get_one_hot_labels(labels.to(self.device), self.classes)
                image_one_hot_labels = one_hot_labels[:, :, None, None]
                image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.image_size, self.image_size)

                self.disc_optim.zero_grad()
                noise = get_uniform_noise((curr_batch, self.noise_channels), device=self.device)
                noise_and_labels = concat_vectors(noise, one_hot_labels)
                fake_images = self.gen(noise_and_labels)
                print(fake_images.shape)
                assert len(fake_images) == len(real_images)
                assert tuple(noise_and_labels.shape) == (curr_batch, noise.shape[1] + one_hot_labels.shape[1])

                fake_images_and_labels = concat_vectors(fake_images.detach(), image_one_hot_labels)
                real_images_and_labels = concat_vectors(real_images, image_one_hot_labels)
                fake_pred = self.disc(fake_images_and_labels)
                real_pred = self.disc(real_images_and_labels)

                disc_fake_loss = loss_crit(fake_pred, torch.zeros_like(fake_pred))
                disc_real_loss = loss_crit(real_pred, torch.ones_like(real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()
                disc_losses.append(disc_loss.item())

                self.gen_optim.zero_grad()
                fake_image_and_labels = concat_vectors(fake_images, image_one_hot_labels)
                fake_pred = self.disc(fake_image_and_labels)

                gen_loss = loss_crit(fake_pred, torch.ones_like(fake_pred))
                gen_loss.backward()
                self.gen_optim.step()
                gen_losses.append(gen_loss.item())

            self._compute_results_per_epoch(epoch, num_epochs_to_show, fake_images, real_images, gen_losses,
                                            disc_losses, gen_mean_losses, disc_mean_losses, show_losses)
        self._display_results_of_training(epochs, start_time, gen_mean_losses, disc_mean_losses)
