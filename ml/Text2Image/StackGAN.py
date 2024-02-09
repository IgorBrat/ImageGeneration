import time
import datetime
import os

import torch
from torch import Tensor
from torchvision import transforms
from tqdm.auto import tqdm

from ml.GAN.GAN import GAN
from ml.utils.visual import show_images_norm
from ml.utils.tensor_logic import get_uniform_noise, get_normal_noise
from ml.utils.network_misc import ResBlock
from ml.utils.text_processing import preprocess_attributes, ConditionAugmentation, TextEncoder
from ml.utils.losses import get_kld_loss
import matplotlib.pyplot as plt


class Stage1Gen(torch.nn.Module):
    def __init__(self, noise_channels: int, out_channels: int, text_embedding_dim: int,
                 text_embedding_latent: int) -> None:
        super(Stage1Gen, self).__init__()
        self.noise_channels = noise_channels
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent
        self.cond_aug = ConditionAugmentation(text_embedding_dim=text_embedding_dim,
                                              text_embedding_latent=text_embedding_latent)
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
        text_embedding_reparameterised, embed_mean, embed_log_var = self.cond_aug(text_embedding)
        assert len(noise) == len(text_embedding_reparameterised)
        noise = noise.view(len(noise), self.noise_channels, 1, 1)
        text_embedding_reparameterised = text_embedding_reparameterised.view(len(text_embedding_reparameterised),
                                                                             self.text_embedding_latent, 1, 1)
        noise_with_text_embed = torch.cat((noise, text_embedding_reparameterised), dim=1)
        return self.model(noise_with_text_embed), embed_mean, embed_log_var


class Stage1Disc(torch.nn.Module):
    def __init__(self, input_dim: int, text_embedding_dim: int, text_embedding_latent: int) -> None:
        super(Stage1Disc, self).__init__()
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent
        self.cond_aug = ConditionAugmentation(False, text_embedding_dim=text_embedding_dim,
                                              text_embedding_latent=text_embedding_latent)
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
        text_embedding_lowdim = self.cond_aug(text_embedding)
        text_embedding_lowdim = text_embedding_lowdim.view(len(text_embedding), self.text_embedding_latent, 1,
                                                           1).repeat(1, 1, 4, 4)
        noise_with_text_embed = torch.cat((cnn_out, text_embedding_lowdim), dim=1)

        return self.out(noise_with_text_embed), cnn_out


class Stage2Gen(torch.nn.Module):
    def __init__(self, input_channels: int, out_channels: int, text_embedding_dim: int,
                 text_embedding_latent: int) -> None:
        super(Stage2Gen, self).__init__()
        self.input_channels = input_channels
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent

        self.cond_aug = ConditionAugmentation(text_embedding_dim=text_embedding_dim,
                                              text_embedding_latent=text_embedding_latent)
        self.down_sample = torch.nn.Sequential(  # 64x64
            self.conv_block(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            self.conv_block(64, 256, kernel_size=3, stride=2, padding=1),  # 16x16
        )
        self.up_sample = torch.nn.Sequential(  # 16x16
            self.deconv_block(256 + text_embedding_latent, 64, kernel_size=5, stride=2, padding=2),  # 8x8
            ResBlock(64),
            ResBlock(64),
            self.deconv_block(64, 32, kernel_size=5, stride=2, padding=2),  # 64x64
            ResBlock(32),
            ResBlock(32),
            self.deconv_block(32, 16, kernel_size=5, stride=2, padding=2),  # 128x128
            ResBlock(16),
            ResBlock(16),
            self.deconv_block(16, out_channels, kernel_size=5, stride=2, padding=2, is_last=True),  # 256x256
        )

    @staticmethod
    def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                   padding: int = 0, leak: float = 0.2):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(leak),
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
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 1),
                torch.nn.Tanh(),
            )

    def forward(self, low_res_imgs, text_embedding: Tensor):
        down_sampled_imgs = self.down_sample(low_res_imgs)
        text_embedding = text_embedding.view(len(text_embedding), self.text_embedding_channels)
        text_embedding_reparameterised, embed_mean, embed_log_var = self.cond_aug(text_embedding)
        text_embedding_reparameterised = (
            text_embedding_reparameterised.view(len(text_embedding), self.text_embedding_latent, 1, 1)
            .repeat(1, 1, down_sampled_imgs.shape[-2], down_sampled_imgs.shape[-1]))
        imgs_with_text_embed = torch.cat((down_sampled_imgs, text_embedding_reparameterised), dim=1)
        return self.up_sample(imgs_with_text_embed), embed_mean, embed_log_var


class Stage2Disc(torch.nn.Module):
    def __init__(self, image_channels: int, text_embedding_dim: int, text_embedding_latent: int) -> None:
        super(Stage2Disc, self).__init__()
        self.text_embedding_channels = text_embedding_dim
        self.text_embedding_latent = text_embedding_latent

        self.cond_aug = ConditionAugmentation(False, text_embedding_dim=text_embedding_dim,
                                              text_embedding_latent=text_embedding_latent)
        self.down_sample = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),  # 128x128
            ResBlock(16),
            ResBlock(16),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            self.pooling_block(kernel_size=2, stride=2),  # 64x64
            ResBlock(32),
            ResBlock(32),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            self.pooling_block(kernel_size=2, stride=2),  # 32x32
            ResBlock(64),
            ResBlock(64),
            torch.nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            self.pooling_block(kernel_size=2, stride=2),  # 16x16
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),  # 8x8
            ResBlock(64),
            ResBlock(64),
            self.pooling_block(kernel_size=2, stride=2),  # 4x4
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear((64 + text_embedding_latent) * 4 * 4, 128),
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

    def forward(self, high_res_imgs, text_embedding: Tensor):
        down_sampled_imgs = self.down_sample(high_res_imgs)
        text_embedding = text_embedding.view(len(text_embedding), self.text_embedding_channels)
        text_embedding_lowdim = self.cond_aug(text_embedding)
        text_embedding_lowdim = (text_embedding_lowdim.view(len(text_embedding), self.text_embedding_latent, 1, 1)
                                 .repeat(1, 1, down_sampled_imgs.shape[-2], down_sampled_imgs.shape[-1]))
        imgs_with_text_embed = torch.cat((down_sampled_imgs, text_embedding_lowdim), dim=1)
        return self.out(imgs_with_text_embed)


class Stage1GAN(GAN):
    """
    Stage1GAN for StackGAN
    Has the architecture of DCGAN
    """

    def __init__(self, noise_dim, image_channels: int = 3, text_embedding_dim: int = 300,
                 text_embedding_latent_dim: int = 128, device: str = 'cpu'):
        super().__init__(noise_dim, image_channels, presaved=False, device=device)
        self.gen = Stage1Gen(noise_dim, image_channels, text_embedding_dim, text_embedding_latent_dim).to(
            device)
        self.disc = Stage1Disc(image_channels, text_embedding_dim, text_embedding_latent_dim).to(device)
        self.init_models()

    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool, show_losses: bool,
              lrn_rate: float, beta1: float, beta2: float):
        """Placeholder for training Stage1GAN independently. Not implemented for now"""
        pass


class Stage2GAN(GAN):
    """
    Stage2GAN for StackGAN
    Has the architecture of DCGAN with residual blocks
    """

    def __init__(self, image_channels: int = 3, text_embedding_dim: int = 300,
                 text_embedding_latent_dim: int = 128, device: str = 'cpu'):
        super().__init__(image_channels, image_channels, presaved=False, device=device)
        self.gen = Stage2Gen(image_channels, image_channels, text_embedding_dim,
                             text_embedding_latent_dim).to(device)
        self.disc = Stage2Disc(image_channels, text_embedding_dim, text_embedding_latent_dim).to(device)
        self.init_models()

    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool, show_losses: bool,
              lrn_rate: float, beta1: float, beta2: float):
        """Placeholder for training Stage2GAN independently. Not implemented for now"""
        pass


class StackGAN:
    """
    StackGAN model, consisting of two GANs which generate images of low and high resolution
    to create higher-quality result
    """

    def __init__(self, noise_channels: int = 100, image_channels: int = 3, text_embedding_latent_channels: int = 128,
                 presaved: bool = False, save_dir: str = None, params_savefile: str = None,
                 text_model: str = "msmarco-distilbert-base-tas-b", device: str = 'cpu'):
        if presaved:
            params = GAN.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            image_channels = params['image_channels']
            text_embedding_latent_channels = params['text_embedding_latent_channels']
            text_model = params['text_model']
        self.text_encoder = TextEncoder(text_model)
        self.text_embedding_latent_channels = text_embedding_latent_channels
        text_embedding_dim = self.text_encoder.get_embedding_dimension()

        self.stage1gan = Stage1GAN(noise_channels, image_channels, text_embedding_dim,
                                   text_embedding_latent_channels, device)
        self.stage2gan = Stage2GAN(image_channels, text_embedding_dim, text_embedding_latent_channels, device)

        self.image_channels = image_channels
        self.noise_channels = noise_channels
        self.image_size_stage1 = 64
        self.image_size_stage2 = 256

        self.trained_epochs = 0

        self.default_savedir = self.stage1gan.default_savedir
        self.stage1gan.default_gen_savefile = 'stage1gen.pt'
        self.stage1gan.default_disc_savefile = 'stage1disc.pt'
        self.stage2gan.default_gen_savefile = 'stage2gen.pt'
        self.stage2gan.default_disc_savefile = 'stage2disc.pt'
        self.default_params_savefile = 'params.json'

        self.transform_stage2 = transforms.Compose([
            transforms.Resize((self.image_size_stage2, self.image_size_stage2)),
        ])

        self.device = device

    def save(self, save_dir: str = None,
             gen1_savefile: str = None, disc1_savefile: str = None,
             gen2_savefile: str = None, disc2_savefile: str = None,
             params_savefile: str = None):
        """
        Save models parameters to specified directory and files
        :param save_dir: save directory
        :param gen1_savefile: save file for Stage1 generator state
        :param disc1_savefile: save file for Stage1 discriminator state
        :param gen2_savefile: save file for Stage2 generator state
        :param disc2_savefile: save file for Stage2 discriminator state
        :param params_savefile: save file for init params
        """
        self.stage1gan.save(save_dir, gen1_savefile, disc1_savefile, save_params=False)
        self.stage2gan.save(save_dir, gen2_savefile, disc2_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'text_embedding_latent_channels': self.text_embedding_latent_channels,
            'text_model': self.text_encoder.model_name,
        }
        GAN._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None,
                          gen1_savefile: str = None, disc1_savefile: str = None,
                          gen2_savefile: str = None, disc2_savefile: str = None,
                          params_savefile: str = None):
        """
        Save models parameters to specified directory and files with training history
        :param save_dir: save directory
        :param gen1_savefile: save file for Stage1 generator state
        :param disc1_savefile: save file for Stage1 discriminator state
        :param gen2_savefile: save file for Stage2 generator state
        :param disc2_savefile: save file for Stage2 discriminator state
        :param params_savefile: save file for init params
        """
        self.stage1gan.save_with_history(save_dir, gen1_savefile, disc1_savefile, save_params=False)
        self.stage2gan.save_with_history(save_dir, gen2_savefile, disc2_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'image_channels': self.image_channels,
            'text_embedding_latent_channels': self.text_embedding_latent_channels,
            'text_model': self.text_encoder.model_name,
        }
        GAN._save_params(save_dir, params_savefile, params)

    def load(self, save_dir: str = None, gen1_savefile: str = None, disc1_savefile: str = None,
             gen2_savefile: str = None, disc2_savefile: str = None):
        """
        Load models state from specified directory and files
        :param save_dir: save directory
        :param gen1_savefile: save file with Stage1 generator state
        :param disc1_savefile: save file with Stage1 discriminator state
        :param gen2_savefile: save file with Stage2 generator state
        :param disc2_savefile: save file with Stage2 discriminator state
        """
        self.stage1gan.load(save_dir, gen1_savefile, disc1_savefile)
        self.stage2gan.load(save_dir, gen2_savefile, disc2_savefile)

    def eval(self):
        """
        Set model to evaluation mode
        """
        self.stage1gan.eval()
        self.stage2gan.eval()

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
        generated, *_ = self.stage1gan.gen(noise, query_emb)
        generated, *_ = self.stage2gan.gen(generated, query_emb)
        return generated

    def train(self, dataloader, epochs: int, num_epochs_to_show: int, show_progress: bool = True,
              show_losses: bool = True, lrn_rate1: float = 1e-4, lrn_rate2: float = 1e-4, beta1_1: float = 0.9,
              beta2_1: float = 0.999, beta1_2: float = 0.9, beta2_2: float = 0.999):
        """
        Train method
        :param dataloader: Dataloader object containing dataset
        :param epochs: number of epochs to train model
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :param show_losses: bool value indicating if losses have to be shown
        :param lrn_rate1: Stage1 optimiser learning rate
        :param lrn_rate2: Stage1 optimiser learning rate
        :param beta1_1: Stage1 beta1 for Adam optimiser
        :param beta2_1: Stage1 beta2 for Adam optimiser
        :param beta1_2: Stage2 beta1 for Adam optimiser
        :param beta2_2: Stage2 beta2 for Adam optimiser
        :return:
        """
        self._prepare_for_training(lrn_rate1, lrn_rate2, beta1_1, beta2_1, beta1_2, beta2_2)

        # Save logs
        start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(os.getcwd(), r'ml/logs/', f'stackgan_{start}.txt')
        with open(log_file, 'w') as file:
            file.write(f'Model: StackGAN\nStart of training: {start}\n')

        loss_crit = torch.nn.BCELoss()

        disc1_losses = []
        gen1_losses = []
        disc2_losses = []
        gen2_losses = []
        disc1_mean_losses = []
        gen1_mean_losses = []
        disc2_mean_losses = []
        gen2_mean_losses = []

        if show_progress:
            dataloader = tqdm(dataloader)

        start_epoch = self.trained_epochs if self.trained_epochs else 1
        start_time = time.time()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, epochs + start_epoch):
            for real_images_low_res, labels, attributes in dataloader:
                curr_batch = len(real_images_low_res)
                real_images_low_res = real_images_low_res.to(self.device)
                attributes = preprocess_attributes(attributes)

                text_emb = self.text_encoder.encode(attributes).to(self.device)

                # ============ STAGE I DISCRIMINATOR ============
                self.stage1gan.disc_optim.zero_grad()
                noise = get_uniform_noise((curr_batch, self.stage1gan.noise_channels, 1, 1), self.device)
                fake_images_low_res, mean, log_var = self.stage1gan.gen(noise, text_emb)
                assert len(fake_images_low_res) == len(real_images_low_res)

                fake_pred_1, _ = self.stage1gan.disc(fake_images_low_res.detach(), text_emb)
                real_pred_1, _ = self.stage1gan.disc(real_images_low_res, text_emb)

                disc1_fake_loss = loss_crit(fake_pred_1, torch.zeros_like(fake_pred_1))
                disc1_real_loss = loss_crit(real_pred_1, torch.ones_like(real_pred_1))
                disc1_loss = (disc1_fake_loss + disc1_real_loss) / 2
                disc1_loss.backward(retain_graph=True)
                self.stage1gan.disc_optim.step()
                disc1_losses.append(disc1_loss.item())

                # ============ STAGE I GENERATOR ============
                self.stage1gan.gen_optim.zero_grad()
                fake_pred_1, _ = self.stage1gan.disc(fake_images_low_res, text_emb)
                gen1_loss = loss_crit(fake_pred_1, torch.ones_like(fake_pred_1)) + get_kld_loss(mean, log_var)
                gen1_loss.backward()
                self.stage1gan.gen_optim.step()
                gen1_losses.append(gen1_loss.item())

                # ============ STAGE II Discriminator ============
                self.stage2gan.disc_optim.zero_grad()
                # Generating again cause variables were already freed
                fake_images_low_res, mean, log_var = self.stage1gan.gen(noise, text_emb)
                real_images_high_res = self.transform_stage2(real_images_low_res)
                fake_images_high_res, mean, log_var = self.stage2gan.gen(fake_images_low_res, text_emb)
                assert fake_images_high_res.shape == real_images_high_res.shape

                fake_pred_2 = self.stage2gan.disc(fake_images_high_res.detach(), text_emb)
                real_pred_2 = self.stage2gan.disc(real_images_high_res, text_emb)

                disc2_fake_loss = loss_crit(fake_pred_2, torch.zeros_like(fake_pred_2))
                disc2_real_loss = loss_crit(real_pred_2, torch.ones_like(real_pred_2))
                disc2_loss = (disc2_fake_loss + disc2_real_loss) / 2
                disc2_loss.backward(retain_graph=True)
                self.stage2gan.disc_optim.step()
                disc2_losses.append(disc2_loss.item())

                # ============ STAGE II GENERATOR ============
                self.stage2gan.gen_optim.zero_grad()
                fake_pred_2 = self.stage2gan.disc(fake_images_high_res, text_emb)
                gen2_loss = loss_crit(fake_pred_2, torch.ones_like(fake_pred_2)) + get_kld_loss(mean, log_var)
                gen2_loss.backward()
                self.stage2gan.gen_optim.step()
                gen2_losses.append(gen2_loss.item())

            self._compute_results_per_epoch_per_stage(epoch, start_epoch, num_epochs_to_show, fake_images_low_res,
                                                      real_images_low_res, gen1_losses, disc1_losses, gen1_mean_losses,
                                                      disc1_mean_losses, show_losses, stage=1)
            self._compute_results_per_epoch_per_stage(epoch, start_epoch, num_epochs_to_show, fake_images_high_res,
                                                      real_images_high_res, gen2_losses, disc2_losses, gen2_mean_losses,
                                                      disc2_mean_losses, show_losses, stage=2)
            with open(log_file, 'a') as file:
                file.write(f'Epoch {epoch}, '
                           f'Gen1 Loss: {gen1_mean_losses[-1]:.4f}, '
                           f'Disc1 Loss: {disc1_mean_losses[-1]:.4f}\n'
                           f'Gen2 Loss: {gen2_mean_losses[-1]:.4f}, '
                           f'Disc2 Loss: {disc2_mean_losses[-1]:.4f}\n'
                           )
        duration = (time.time() - start_time) / 60
        print(f'Finished training after {epochs} epochs and', end=' ')
        if duration > 60:
            print(f'{round(duration / 60, 2)} h')
        else:
            print(f'{round(duration, 2)} m')
        self._display_results_of_training_per_stage(gen1_mean_losses, disc1_mean_losses, 1)
        self._display_results_of_training_per_stage(gen2_mean_losses, disc2_mean_losses, 2)
        self.eval()
        self.trained_epochs += epochs
        self.stage1gan.trained_epochs += epochs
        self.stage2gan.trained_epochs += epochs
        with open(log_file, 'a') as file:
            file.write(f'Finished training in {round(duration, 2)} min')

    def _prepare_for_training(self, lrn_rate1, lrn_rate2, beta1_1, beta2_1, beta1_2, beta2_2):
        """
        Util method for preparing model and optimiser for training
        :param lrn_rate1: Stage1 optimiser learning rate
        :param lrn_rate2: Stage1 optimiser learning rate
        :param beta1_1: Stage1 beta1 for Adam optimiser
        :param beta2_1: Stage1 beta2 for Adam optimiser
        :param beta1_2: Stage2 beta1 for Adam optimiser
        :param beta2_2: Stage2 beta2 for Adam optimiser
        """
        self.stage1gan._prepare_for_training(lrn_rate1, beta1_1, beta2_1)
        self.stage2gan._prepare_for_training(lrn_rate2, beta1_2, beta2_2)

    def _display_losses_per_stage(self, gen_mean_losses, disc_mean_losses, epoch, start_epoch, stage: int):
        """
        Display generator and discriminator losses of specified stage
        :param gen_mean_losses: mean losses of generator
        :param disc_mean_losses: mean losses of discriminator
        :param epoch: current epoch
        :param start_epoch: starting epoch
        :param stage: current stage: 1 or 2
        """
        plt.plot(
            range(start_epoch, epoch + start_epoch),
            Tensor(gen_mean_losses),
            label=f"Stage {stage} Generator Loss"
        )
        plt.plot(
            range(start_epoch, epoch + start_epoch),
            Tensor(disc_mean_losses),
            label=f"Stage {stage} Discriminator Loss"
        )
        plt.legend()
        plt.show()

    def _compute_results_per_epoch_per_stage(self, epoch, start_epoch, num_epochs_to_show, fake_images, real_images,
                                             gen_losses, disc_losses, gen_mean_losses, disc_mean_losses, show_losses,
                                             stage):
        """
        Compute intermediate losses and display current results of training of specified stage
        :param epoch: current epoch
        :param start_epoch: starting epoch
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param fake_images: last generated images batch
        :param real_images: last real images batch
        :param gen_losses: losses of generator
        :param disc_losses: losses of discriminator
        :param gen_mean_losses: list of mean generator losses
        :param disc_mean_losses: list of mean discriminator losses
        :param show_losses: bool value indicating if losses have to be shown
        :param stage: current stage: 1 or 2
        :return:
        """
        gen_mean_losses.append(sum(gen_losses) / len(gen_losses))
        disc_mean_losses.append(sum(disc_losses) / len(disc_losses))
        gen_losses.clear()
        disc_losses.clear()
        if show_losses and num_epochs_to_show and (not epoch % num_epochs_to_show):
            show_images_norm(fake_images, num=9)
            show_images_norm(real_images, num=9)
            self._display_losses_per_stage(gen_mean_losses, disc_mean_losses, epoch, start_epoch, stage=stage)

    def _display_results_of_training_per_stage(self, gen_mean_losses, disc_mean_losses, stage: int):
        """
        Util method for displaying results of training of specified stage
        :param gen_mean_losses: mean losses of generator
        :param disc_mean_losses: mean losses of discriminator
        :param stage: current stage: 1 or 2
        :return:
        """
        print(f'Mean stage {stage} generator loss: {round(sum(gen_mean_losses) / len(gen_mean_losses), 4)}')
        print(f'Mean stage {stage} discriminator loss: {round(sum(disc_mean_losses) / len(disc_mean_losses), 4)}')
        gen_mean_losses.clear()
        disc_mean_losses.clear()
