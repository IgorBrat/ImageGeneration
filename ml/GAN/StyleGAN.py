import torch
from math import log2
from tqdm.auto import tqdm
from ml.utils.tensor_logic import get_normal_noise
from ml.utils.visual import show_images_norm
from ml.utils.data_management import load_data
from ml.GAN.GAN import GAN
from ml.GAN.misc.Style.AdaIN import AdaIN
from ml.GAN.misc.Style.MLP import MultiLayerPerceptron
from ml.GAN.misc.Style.ScaledLayers import WeightedScaledConvo
from ml.GAN.misc.Style.WeightedNoise import InjectWeightedNoise
from ml.GAN.misc.Style.Blocks import GeneratorBlock, Conv2Block


class Generator(torch.nn.Module):
    def __init__(self, z_chan, w_chan, in_chan, channel_factors, out_img_chan: int = 3, device: str = 'cpu'):
        super(Generator, self).__init__()
        self.starting_const = torch.nn.Parameter(torch.ones((1, in_chan, 4, 4)))
        self.mlp = MultiLayerPerceptron(z_chan, w_chan)
        self.init_adain1 = AdaIN(in_chan, w_chan)
        self.init_adain2 = AdaIN(in_chan, w_chan)
        self.init_noise1 = InjectWeightedNoise(in_chan, device)
        self.init_noise2 = InjectWeightedNoise(in_chan, device)
        self.init_conv = torch.nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1)
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=True)

        self.init_rgb = WeightedScaledConvo(
            in_chan, out_img_chan, kernel=1, stride=1, padding=0
        )
        self.progressive_blocks, self.rgb_layers = (
            torch.nn.ModuleList([]),
            torch.nn.ModuleList([self.init_rgb]),
        )

        for idx in range(len(channel_factors) - 1):
            conv_in_c = int(in_chan * channel_factors[idx])
            conv_out_c = int(in_chan * channel_factors[idx + 1])
            self.progressive_blocks.append(GeneratorBlock(conv_in_c, conv_out_c, w_chan))
            self.rgb_layers.append(WeightedScaledConvo(conv_out_c, out_img_chan, kernel=1, padding=0))

    @staticmethod
    def fade_in(alpha, up_scaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * up_scaled)

    def forward(self, z, alpha, num_steps):
        w = self.mlp(z)
        feat = self.init_adain1(self.init_noise1(self.starting_const), w)
        feat = self.init_conv(feat)
        curr_out = self.init_adain2(self.lrelu(self.init_noise2(feat)), w)

        if not num_steps:
            return self.init_rgb(feat)

        for step in range(num_steps):
            up_scaled = torch.nn.functional.interpolate(curr_out, scale_factor=2, mode="bilinear")
            curr_out = self.progressive_blocks[step](up_scaled, w)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively

        final_up_scaled = self.rgb_layers[num_steps - 1](up_scaled)
        final_out = self.rgb_layers[num_steps](curr_out)
        return self.fade_in(alpha, final_up_scaled, final_out)


class Discriminator(torch.nn.Module):
    def __init__(self, in_chan: int, channel_factors, img_chan: int = 3):
        super(Discriminator, self).__init__()
        self.down_sample_blocks, self.rgb_layers = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        for idx in range(len(channel_factors) - 1, 0, -1):
            in_conv_chan = int(in_chan * channel_factors[idx])
            out_conv_chan = int(in_chan * channel_factors[idx - 1])
            self.down_sample_blocks.append(Conv2Block(in_conv_chan, out_conv_chan))
            self.rgb_layers.append(WeightedScaledConvo(img_chan, in_conv_chan, kernel=1, padding=0))
        self.lrelu = torch.nn.LeakyReLU(0.2)
        # grb layer for 4x4 input images
        self.init_rgb_layer = WeightedScaledConvo(img_chan, in_chan, kernel=1, padding=0)
        self.rgb_layers.append(self.init_rgb_layer)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        # block for 4x4 input image
        self.final_block = torch.nn.Sequential(
            WeightedScaledConvo(in_chan + 1, in_chan),
            torch.nn.LeakyReLU(0.2),
            WeightedScaledConvo(in_chan, in_chan, kernel=4, padding=0),
            torch.nn.LeakyReLU(0.2),
            WeightedScaledConvo(in_chan, 1, kernel=1, padding=0),
        )

    @staticmethod
    def pro_growth(alpha, down_scaled_output, model_output):
        return alpha * model_output + (1 - alpha) * down_scaled_output

    @staticmethod
    def minibatch_std(feat):
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        batch_std = torch.std(feat, dim=0).mean().repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
        return torch.cat([feat, batch_std], dim=1)

    def forward(self, feat, alpha, num_steps):
        # num of steps is defining what levels of 4x4-1024x1024 we want to pass in discriminator
        init_step = len(self.down_sample_blocks) - num_steps
        # convert from rgb
        curr_out = self.lrelu(self.rgb_layers[init_step](feat))
        if not num_steps:  # image is 4x4
            curr_out = self.minibatch_std(curr_out)
            return self.final_block(curr_out).view(curr_out.shape[0], -1)

        # because downscale blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.lrelu(self.rgb_layers[init_step + 1](self.avg_pool(feat)))
        curr_out = self.avg_pool(self.down_sample_blocks[init_step](curr_out))

        curr_out = self.pro_growth(alpha, downscaled, curr_out)

        for step in range(init_step + 1, len(self.down_sample_blocks)):
            curr_out = self.avg_pool(self.down_sample_blocks[step](curr_out))

        curr_out = self.minibatch_std(curr_out)
        return self.final_block(curr_out).view(curr_out.shape[0], -1)


class StyleGAN(GAN):
    def __init__(self, noise_channels: int = 256, latent_space_dim: int = 256, input_channels: int = 256,
                 output_image_channels: int = 3, init_image_size: int = 4, presaved: bool = False, save_dir: str = None,
                 params_savefile: str = None, device: str = 'cpu'):
        if presaved:
            params = self.load_params(save_dir, params_savefile)
            noise_channels = params['noise_channels']
            latent_space_dim = params['latent_space_dim']
            input_channels = params['input_image_channels']
            output_image_channels = params['output_image_channels']
            init_image_size = params['init_image_size']
        super().__init__(noise_channels, output_image_channels, presaved=presaved, device=device)
        self.latent_space_dim = latent_space_dim
        self.input_channels = input_channels
        self.init_image_size = init_image_size
        self.channel_factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
        self.batch_sizes = [256, 128, 64, 32, 16, 8]
        self.prog_epochs = [20] * len(self.batch_sizes)
        self.gen = Generator(z_chan=noise_channels, w_chan=latent_space_dim, in_chan=input_channels,
                             out_img_chan=output_image_channels, channel_factors=self.channel_factors).to(device)
        self.disc = Discriminator(in_chan=input_channels, img_chan=output_image_channels,
                                  channel_factors=self.channel_factors)

    def save(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
             params_savefile: str = None):
        super().save(save_dir, gen_savefile, disc_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        print(params_savefile)
        params = {
            'noise_channels': self.noise_channels,
            'latent_space_dim': self.latent_space_dim,
            'input_image_channels': self.input_channels,
            'output_image_channels': self.image_channels,
            'init_image_size': self.init_image_size,
        }
        self._save_params(save_dir, params_savefile, params)

    def save_with_history(self, save_dir: str = None, gen_savefile: str = None, disc_savefile: str = None,
                          params_savefile: str = None):
        super().save_with_history(save_dir, gen_savefile, disc_savefile, save_params=False)

        save_dir = save_dir if save_dir else self.default_savedir
        params_savefile = params_savefile if params_savefile else self.default_params_savefile
        params = {
            'noise_channels': self.noise_channels,
            'latent_space_dim': self.latent_space_dim,
            'input_image_channels': self.input_channels,
            'output_image_channels': self.image_channels,
            'init_image_size': self.init_image_size,
        }
        self._save_params(save_dir, params_savefile, params)

    def generate(self, num_steps, num_images: int = 20, alpha: float = 1):
        """
        Generate images
        :param num_steps: number of up-scaling steps
        :param num_images: number of generated images
        :param alpha: controls interpolation between resolution
        :return: generated images
        """
        self.gen.eval()
        with torch.no_grad():
            noise = get_normal_noise((num_images, self.noise_channels), self.device)
            gen_images = self.gen(noise, alpha, num_steps)
        return gen_images

    def gradient_penalty(self, real_images, fake_images, alpha, train_step):
        """
        Calculate gradient penalty
        :param real_images: real images
        :param fake_images: fake (generated) images
        :param alpha: controls interpolation between resolution
        :param train_step: number of down-scaling steps
        :return: gradient penalty
        """
        batch, chan, height, width = real_images.shape
        beta = get_normal_noise((batch, 1, 1, 1), self.device).repeat(1, chan, height, width)
        interpolated_images = real_images * beta + fake_images.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.disc(interpolated_images, alpha, train_step)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2)

    def train_step(self, dataloader, dataset_len, step, alpha, lambda_grad_penalty, show_progress: bool = True):
        """
        Do one train step
        :param dataloader: Dataloader object containing dataset
        :param dataset_len: number of images in dataset
        :param step: current step
        :param alpha: controls interpolation between resolution
        :param lambda_grad_penalty: coefficient of gradient penalty
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :return: updated alpha
        """
        if show_progress:
            dataloader = tqdm(dataloader, leave=True)

        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            cur_batch_size = real_images.shape[0]

            noise = get_normal_noise((cur_batch_size, self.noise_channels), self.device)

            fake = self.gen(noise, alpha, step)
            critic_real = self.disc(real_images, alpha, step)
            critic_fake = self.disc(fake.detach(), alpha, step)
            grad_pen = self.gradient_penalty(real_images, fake, alpha, step)
            loss_disc = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + lambda_grad_penalty * grad_pen
                    + (1e-3 * torch.mean(critic_real ** 2))
            )

            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()

            gen_fake = self.disc(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

            self.gen_optim.zero_grad()
            loss_gen.backward()
            self.gen_optim.step()

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / ((self.prog_epochs[step] * 0.5) * dataset_len)
            alpha = min(alpha, 1)

            if show_progress:
                dataloader.set_postfix(
                    gradient_penalty=grad_pen.item(),
                    loss_gen=loss_gen.item(),
                    loss_disc=loss_disc.item(),
                )
        return alpha

    def train(self, datadir: str, epochs: int, num_epochs_to_show: int, show_progress: bool = True,
              lrn_rate: float = 1e-3, lrn_rate_mlp: float = 1e-5, beta1: float = 0.9,
              beta2: float = 0.999, alpha: float = 1e-5, lambda_grad_penalty: float = 10):
        """
        :param datadir: directory containing dataset
        :param epochs: number of epochs to train model for
        :param num_epochs_to_show: number of epochs to show generated images and losses for debugging
        :param show_progress: bool value indicating if tqdm progress has to be shown
        :param lrn_rate: optimiser learning rate
        :param lrn_rate_mlp: mlp optimiser learning rate
        :param beta1: param of the same name for Adam optimiser
        :param beta2: param of the same name for Adam optimiser
        :param alpha: controls interpolation between resolution
        :param lambda_grad_penalty: coefficient of gradient penalty
        :return:
        """
        self.gen_optim = torch.optim.Adam([
            {"params": [param for name, param in self.gen.named_parameters() if "mlp" not in name]},
            {"params": self.gen.mlp.parameters(), "lr": lrn_rate_mlp}],
            lr=lrn_rate,
            betas=(beta1, beta2),
        )
        self.disc_optim = torch.optim.Adam(
            self.disc.parameters(), lr=lrn_rate, betas=(beta1, beta2),
        )

        self.gen.train()
        self.disc.train()
        self.count_parameters()

        step = int(log2(self.init_image_size / 4))
        for num_epochs in self.prog_epochs[step:]:
            # start with very low alpha
            loader, dataset = load_data(datadir, 4 * 2 ** step)
            print(f"Current image size: {4 * 2 ** step}")

            for epoch in range(num_epochs):
                print(f"Epoch [{epoch + 1}/{num_epochs}]")
                alpha = self.train_step(
                    loader,
                    len(dataset),
                    step,
                    alpha,
                    lambda_grad_penalty,
                )

            imgs = self.generate_examples(step, 20, alpha)
            show_images_norm(imgs, num=imgs.shape[0], num_row=int(imgs.shape[0] ** 0.5))

            step += 1  # progress to the next img size
