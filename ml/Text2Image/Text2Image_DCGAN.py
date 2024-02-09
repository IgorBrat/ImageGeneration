import time
import os
import datetime
import torch
from torch import Tensor
from tqdm.auto import tqdm
from ml.utils.tensor_logic import get_uniform_noise, get_normal_noise
from ml.utils.text_processing import preprocess_attributes, TextEncoder
from ml.GAN.GAN import GAN


class Generator(torch.nn.Module):
    def __init__(self, noise_channels: int, out_channels: int = 3, text_embedding_dim: int = 300,
                 text_embedding_latent: int = 128) -> None:
        super(Generator, self).__init__()
        self.noise_channels = noise_channels
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent
        self.text_embed = torch.nn.Sequential(
            torch.nn.Linear(text_embedding_dim, text_embedding_latent),
            torch.nn.BatchNorm1d(text_embedding_latent),
            torch.nn.LeakyReLU(0.2),
        )
        self.model = torch.nn.Sequential(
            self.gen_block(noise_channels + text_embedding_latent, 256),
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

    def forward(self, noise, text_embedding: Tensor):
        text_embedding = text_embedding.view(len(text_embedding), self.text_embedding_channels)
        text_embedding_lowdim = self.text_embed(text_embedding)
        assert len(noise) == len(text_embedding)
        noise = noise.view(len(noise), self.noise_channels, 1, 1)
        text_embedding_lowdim = text_embedding_lowdim.view(len(text_embedding_lowdim), self.text_embedding_latent, 1, 1)
        noise_with_text_embed = torch.cat((noise, text_embedding_lowdim), dim=1)
        return self.model(noise_with_text_embed)


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim: int, text_embedding_dim: int, text_embedding_latent: int) -> None:
        super(Discriminator, self).__init__()
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent
        self.text_embed = torch.nn.Sequential(
            torch.nn.Linear(text_embedding_dim, text_embedding_latent),
            torch.nn.BatchNorm1d(text_embedding_latent),
            torch.nn.LeakyReLU(0.2),
        )
        self.model = torch.nn.Sequential(
            self.disc_block(input_dim, 32, stride=2, padding=2),
            self.disc_block(32, 64, stride=2, padding=2),
            self.disc_block(64, 128, stride=2, padding=2),
            self.disc_block(128, 256, stride=2, padding=2),
        )
        self.out = self.disc_block(256 + text_embedding_latent, 1, kernel_size=4, is_last=True)

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

    def forward(self, noise, text_embedding):
        text_embedding = text_embedding.view(len(text_embedding), self.text_embedding_channels)
        assert len(noise) == len(text_embedding)
        cnn_out = self.model(noise)
        text_embedding_lowdim = self.text_embed(text_embedding)
        text_embedding_lowdim = text_embedding_lowdim.view(len(text_embedding), self.text_embedding_latent, 1,
                                                           1).repeat(1, 1, 4, 4)
        noise_with_text_embed = torch.cat((cnn_out, text_embedding_lowdim), dim=1)

        return self.out(noise_with_text_embed), cnn_out


class Text2ImageDCGAN(GAN):
    """
    DCGAN generating images given text queries
    """

    def __init__(self, noise_channels: int = 100, image_channels: int = 3, text_embedding_latent_channels: int = 128,
                 text_model: str = "msmarco-distilbert-base-tas-b", presaved: bool = False,
                 save_dir: str = None, params_savefile: str = None, device: str = 'cpu'):
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            image_channels = params['image_channels']
            text_embedding_latent_channels = params['text_embedding_latent_channels']
            text_model = params['text_model']
        super().__init__(noise_channels, image_channels, presaved, device=device)
        self.text_encoder = TextEncoder(text_model)
        self.text_embedding_latent_channels = text_embedding_latent_channels
        text_embedding_dim = self.text_encoder.get_embedding_dimension()
        self.gen = Generator(noise_channels, image_channels, text_embedding_dim,
                             text_embedding_latent_channels).to(device)
        self.disc = Discriminator(image_channels, text_embedding_dim,
                                  text_embedding_latent_channels).to(device)

        self.init_models()

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
            'text_embedding_latent_channels': self.text_embedding_latent_channels,
            'text_model': self.text_encoder.model_name,
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
            'text_embedding_latent_channels': self.text_embedding_latent_channels,
            'text_model': self.text_encoder.model_name,
        }
        self._save_params(save_dir, params_savefile, params)

    def generate(self, query: str, num: int = 9, distribution: str = 'normal'):
        """
        Generate images given text prompt
        :param query: text query
        :param num: number of generated images
        :param distribution: input noise distribution, one of ['normal', 'uniform']
        :return: generated images
        """
        self.eval()
        query_emb = self.text_encoder.encode([query]).to(self.device)
        query_emb = query_emb.repeat(num, 1, 1)
        if distribution == 'normal':
            noise = get_normal_noise((num, self.noise_channels, 1, 1), self.device)
        elif distribution == 'uniform':
            noise = get_uniform_noise((num, self.noise_channels, 1, 1), self.device)
        else:
            raise ValueError('Only "normal" and "uniform" distributions are supported for noise')
        return self.gen(noise, query_emb)

    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool = True,
              show_losses: bool = True, lrn_rate: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999):
        self._prepare_for_training(lrn_rate, beta1, beta2)

        # Save logs
        start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(os.getcwd(), r'ml/logs/', f'text-dcgan_{start}.txt')
        with open(log_file, 'w') as file:
            file.write(f'Model: Text2Image DCGAN\nStart of training: {start}\n')

        loss_crit = torch.nn.BCELoss()

        gen_losses = []
        disc_losses = []
        gen_mean_losses = []
        disc_mean_losses = []

        if show_progress:
            dataloader = tqdm(dataloader)

        start_epoch = self.trained_epochs if self.trained_epochs else 1
        start_time = time.time()
        for epoch in range(start_epoch, epochs + start_epoch):
            for real_images, labels, attributes in dataloader:
                curr_batch = len(real_images)
                real_images = real_images.to(self.device)
                attributes = preprocess_attributes(attributes)

                text_emb = self.text_encoder.encode(attributes).to(self.device)

                # DISCRIMINATOR
                self.disc_optim.zero_grad()
                noise = get_uniform_noise((curr_batch, self.noise_channels, 1, 1), self.device)
                fake_images = self.gen(noise, text_emb)
                assert len(fake_images) == len(real_images)

                fake_pred, _ = self.disc(fake_images.detach(), text_emb)
                real_pred, _ = self.disc(real_images, text_emb)

                disc_fake_loss = loss_crit(fake_pred, torch.zeros_like(fake_pred))
                disc_real_loss = loss_crit(real_pred, torch.ones_like(real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()
                disc_losses.append(disc_loss.item())

                # GENERATOR
                self.gen_optim.zero_grad()
                fake_pred, _ = self.disc(fake_images, text_emb)
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
