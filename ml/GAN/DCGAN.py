import torch
import os
import datetime
import time
from tqdm.auto import tqdm
from ml.utils.tensor_logic import get_uniform_noise
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


class DCGAN(GAN):
    """
    DCGAN model
    """

    def __init__(self, noise_channels: int = 0, image_channels: int = 3, presaved: bool = False, save_dir: str = None,
                 params_savefile: str = None, device: str = 'cpu'):
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            image_channels = params['image_channels']
        super().__init__(noise_channels, image_channels, presaved, device=device)
        self.gen = Generator(noise_channels, image_channels).to(device)
        self.disc = Discriminator(image_channels).to(device)
        self.init_models()

    def train(self, dataloader, epochs, num_epochs_to_show: int = None, show_progress: bool = True,
              show_losses: bool = True, lrn_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self._prepare_for_training(lrn_rate, beta1, beta2)

        # Save logs
        start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(os.getcwd(), r'ml/logs/', f'dcgan_{start}.txt')
        with open(log_file, 'w') as file:
            file.write(f'Model: DCGAN\nStart of training: {start}\n')

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
                curr_batch = len(real_images)
                real_images = real_images.to(self.device)

                self.disc_optim.zero_grad()
                noise = get_uniform_noise((curr_batch, self.noise_channels), self.device)
                fake_images = self.gen(noise)
                assert len(fake_images) == len(real_images)

                fake_pred = self.disc(fake_images)
                real_pred = self.disc(real_images)

                disc_fake_loss = loss_crit(fake_pred, torch.zeros_like(fake_pred))
                disc_real_loss = loss_crit(real_pred, torch.ones_like(real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()
                disc_losses.append(disc_loss.item())

                self.gen_optim.zero_grad()
                fake_pred = self.disc(fake_images)
                gen_loss = loss_crit(fake_pred, torch.ones_like(fake_pred))
                gen_loss.backward()
                self.gen_optim.step()
                gen_losses.append(gen_loss.item())
            self._compute_results_per_epoch(epoch, num_epochs_to_show, fake_images, real_images, gen_losses,
                                            disc_losses, gen_mean_losses, disc_mean_losses, show_losses)
            with open(log_file, 'a') as file:
                file.write(f'Epoch {epoch}, '
                           f'Gen Loss: {gen_mean_losses[-1]:.4f}, '
                           f'Disc Loss: {disc_mean_losses[-1]:.4f}\n'
                           )
        self._display_results_of_training(epochs, start_time, gen_mean_losses, disc_mean_losses)
        with open(log_file, 'a') as file:
            file.write(f'Finished training in {round((time.time() - start_time) / 60, 2)} min')
