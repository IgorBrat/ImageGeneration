import os.path

import torch
from torchvision.models import vgg19
from torchvision.transforms import ToPILImage

from ml.utils.tensor_logic import gram_matrix
from ml.utils.visual import show_images_unnorm


class Normalisation(torch.nn.Module):
    """
    Normalising images while forward propagating
    Mean and Std are chosen respectively to VGG19 normalisation
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalisation, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class NST(torch.nn.Module):
    """
    Network using VGG19 to calculate activations at style and content layers
    """

    def __init__(self, style_layers_indices, model_path: str = None, normalise_image: bool = False):
        super(NST, self).__init__()

        self.style_layers_indices = style_layers_indices
        self.vgg = None
        self.load_vgg(model_path)
        self.normalise_image = normalise_image
        if self.normalise_image:
            self.norm = Normalisation()

    def load_vgg(self, save_dir: str = None):
        """
        Load VGG19 model needed for neural style transfer
        :param save_dir: directory with VGG19 model. If None, model is downloaded and saved to cache
        """
        if save_dir:
            print('Loading saved VGG19 model')
            vgg = vgg19()
            checkpoint = torch.load(save_dir)
            vgg.load_state_dict(checkpoint)
        else:
            print('Downloading VGG19 model and saving to cache')
            vgg = vgg19(weights='DEFAULT', progress=True)
        self.vgg = vgg.features
        self.vgg.eval()
        print('VGG19 loaded successfully.')

    def forward(self, img):
        """
        Calculate activations at desired layers for neural style transfer
        :param img: propagated image
        :return: activations
        """
        activations = []
        if self.normalise_image:
            img = self.norm(img)
        for idx, layer in enumerate(self.vgg):
            img = layer(img)
            if str(idx) in self.style_layers_indices:
                activations.append(img)
        return activations


class StyleTransferModel:
    """
    Model applying neural style transfer on a given content image using style from another one
    """

    def __init__(self, pretrained_vgg_dir: str, style_layers=None, normalise_image: bool = False,
                 normalise_losses: bool = False, device: str = 'cpu'):
        self.style_layers = style_layers if style_layers else \
            [
                ('0', 1),
                ('5', 1),
                ('10', 1),
                ('19', 1),
                ('28', 1),
            ]
        self.style_layers_weights = [style_layer[1] for style_layer in self.style_layers]
        self.normalise_losses = normalise_losses
        self.model = NST(model_path=pretrained_vgg_dir,
                         style_layers_indices=[style_layer[0] for style_layer in self.style_layers],
                         normalise_image=normalise_image).to(device).eval()
        self.device = device

    def calculate_content_cost(self, content_activations: torch.Tensor, generated_activations: torch.Tensor):
        """
        Calculate loss of generated image with regart to content
        :param content_activations: activations of content image
        :param generated_activations: activations of generated image
        :return: loss of generated image with regart to content
        """
        if self.normalise_losses:
            batch, num_channels, height, width = content_activations.shape
            return torch.mean((content_activations - generated_activations) ** 2) / (2 * num_channels * height * width)
        else:
            return torch.mean((content_activations - generated_activations) ** 2) / 2

    def calculate_style_cost(self, style_activations: torch.Tensor, generated_activations: torch.Tensor):
        """
        Calculate loss of generated image with regart to style
        :param style_activations: activations of style image
        :param generated_activations: activations of generated image
        :return: loss of generated image with regart to style
        """
        batch, num_channels, height, width = style_activations.shape
        gram_style = gram_matrix(style_activations.view(num_channels, height * width))
        gram_generated = gram_matrix(generated_activations.view(num_channels, height * width))
        if self.normalise_losses:
            return torch.mean((gram_style - gram_generated) ** 2) / ((2 * num_channels) ** 2 * height * width)
        else:
            return torch.mean((gram_style - gram_generated) ** 2)

    def calculate_cost(self, content_activations, style_activations, generated_activations,
                       alpha: int, beta: int):
        """
        Calculate total loss of generated image
        :param content_activations: activations of content image
        :param style_activations: activations of style image
        :param generated_activations: activations of generated image
        :param alpha: coefficient of content loss
        :param beta: coefficient of style loss
        :return: total loss
        """
        content_cost = style_cost = 0
        for content_act, style_act, generated_act, style_weight in zip(content_activations,
                                                                       style_activations,
                                                                       generated_activations,
                                                                       self.style_layers_weights):
            content_cost += self.calculate_content_cost(content_act, generated_act)
            style_cost += self.calculate_style_cost(style_act, generated_act) * style_weight
        return alpha * content_cost + beta * style_cost

    def mix_style(self, epochs: int, content_img: torch.Tensor, style_img: torch.Tensor, generated_save_dir: str = None,
                  num_epochs_to_show: int = 100, show_loss: bool = True, alpha: int = 1,
                  beta: int = 1e3, lrn_rate: float = 5e-3, beta1: float = 0.9, beta2: float = 0.999):
        """
        Apply style to content image
        :param epochs: number of epochs for applying style
        :param content_img: content image
        :param style_img: style image
        :param generated_save_dir: directory to save generated image
        :param num_epochs_to_show: number of epochs to display progress and losses
        :param show_loss: bool value indicating if losses have to be displayed
        :param alpha: coefficient of content loss
        :param beta: coefficient of style loss
        :param lrn_rate: optimiser learning rate
        :param beta1: param of the same name for Adam optimiser
        :param beta2: param of the same name for Adam optimiser
        :return: mixed image
        """
        # Adding some noise to content image for faster learning
        generated_img = content_img.clone() + (0.3 * torch.rand_like(content_img) - 0.15)
        generated_img.requires_grad_(True)
        optim = torch.optim.Adam([generated_img], lr=lrn_rate, betas=(beta1, beta2))
        for epoch in range(0, epochs + 1):
            generated_features = self.model(generated_img)
            content_feautes = self.model(content_img)
            style_featues = self.model(style_img)

            # iterating over the activation of each layer and
            # calculate the loss and add it to the content and the style loss
            total_loss = self.calculate_cost(content_feautes, style_featues, generated_features, alpha, beta)
            # optimize the pixel values of the generated image and backpropagate the loss
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            with torch.no_grad():
                generated_img.clamp_(0, 1)
            if show_loss and num_epochs_to_show and not epoch % num_epochs_to_show:
                print()
                print(f'Epoch: {epoch}, Loss: {total_loss.item()}')
                show_images_unnorm(generated_img)

        if generated_save_dir:
            transform = ToPILImage()
            generated_img = transform(generated_img)
            generated_img.save(generated_save_dir)
            print(f'Saved generated image to {os.path.join(os.getcwd(), generated_save_dir)}')
        return generated_img
