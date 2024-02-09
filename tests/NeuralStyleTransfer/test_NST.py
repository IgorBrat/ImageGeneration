import os

import numpy as np
import pytest
import torch

from ml.NeuralStyleTransfer.NST import Normalisation, NST, StyleTransferModel


def test_normalisation():
    norm = Normalisation()
    images = torch.randn(10, 3, 64, 64)
    norm_images = norm(images)
    assert norm_images.shape == (10, 3, 64, 64)


class TestNST:
    def test_load(self):
        assert NST(model_path=os.path.join(os.getcwd(), r'ml/resources/vgg19.pt'), style_layers_indices=[('0', 1)])
        with pytest.raises(FileNotFoundError):
            NST(model_path=r'./vgg19.pt', style_layers_indices=['0', '2'])

    def test_propagation(self):
        image = torch.randn(1, 3, 64, 64)
        model = NST(model_path=r'ml/resources/vgg19.pt',
                    style_layers_indices=['0', '5', '10', '19', '28'])
        assert len(model(image)) == 5
        model.style_layers_indices = ['0', '4']
        assert len(model(image)) == 2
        model.style_layers_indices = []
        assert len(model(image)) == 0


class TestStyleTransferModel:
    model = StyleTransferModel(pretrained_vgg_dir=os.path.join(os.getcwd(), r'ml/resources/vgg19.pt'))

    def test_calculate_content_cost(self):
        content_activations = torch.ones(size=(1, 3, 64, 64))
        generated_activations = torch.ones(size=(1, 3, 64, 64)) * 1.5
        self.model.normalise_losses = False
        assert self.model.calculate_content_cost(content_activations, generated_activations) == (1.5 - 1) ** 2 / 2
        self.model.normalise_losses = True
        generated_activations = torch.ones(size=(1, 3, 64, 64)) * 243
        assert (self.model.calculate_content_cost(content_activations, generated_activations) ==
                (243 - 1) ** 2 / (2 * 3 * 64 ** 2))

    def test_calculate_style_cost(self):
        style_activations = torch.ones(size=(1, 3, 2, 2))
        generated_activations = torch.ones(size=(1, 3, 2, 2)) * 2
        self.model.normalise_losses = False
        assert self.model.calculate_style_cost(style_activations, generated_activations) == (4 * (2 ** 2 - 1 ** 2)) ** 2
        self.model.normalise_losses = True
        generated_activations = torch.ones(size=(1, 3, 2, 2)) * 5
        assert (self.model.calculate_style_cost(style_activations, generated_activations) ==
                (4 * (5 ** 2 - 1 ** 2)) ** 2 / ((2 * 3) ** 2 * 2 ** 2))

    def test_calculate_cost(self):
        shape = (1, 3, 2, 2)
        content_activations = [
            torch.ones(size=shape),
            torch.ones(size=shape) * 2,
            torch.ones(size=shape) * 2,
            torch.ones(size=shape) * 3,
            torch.ones(size=shape) * 4,
        ]
        style_activations = [
            torch.ones(size=shape) * 0.5,
            torch.ones(size=shape) * 2,
            torch.ones(size=shape) * 2,
            torch.ones(size=shape) * 4,
            torch.ones(size=shape) * 3.5,
        ]
        generated_activations = [
            torch.ones(size=shape),
            torch.ones(size=shape) * 2,
            torch.ones(size=shape),
            torch.ones(size=shape) * 3,
            torch.ones(size=shape),
        ]
        self.model.normalise_losses = False
        assert not self.model.calculate_cost(content_activations, style_activations, generated_activations,
                                             alpha=0, beta=0)
        assert self.model.calculate_cost(content_activations, style_activations, generated_activations,
                                         alpha=1, beta=0) == 5
        assert self.model.calculate_cost(content_activations, style_activations, generated_activations,
                                         alpha=0, beta=1) == 2962
        self.model.normalise_losses = True
        assert np.isclose(self.model.calculate_cost(content_activations, style_activations, generated_activations,
                                                    alpha=1, beta=1), 20.99, atol=1e-2, rtol=0)
