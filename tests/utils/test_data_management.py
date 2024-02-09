import math

from ml.utils.data_management import (transform_norm, transform_unnorm, load_data, load_cub, Cub2011)
import pytest
import numpy as np
import torch
from torchvision import transforms


class TestTransformNorm:
    image_size = 128
    transform = transform_norm(image_size)
    tensor = torch.rand(3, image_size, image_size)
    image = transforms.ToPILImage()(tensor)

    def test_negative_image_size(self):
        image_size = -5
        with pytest.raises(ValueError, match='Image size must be a positive number'):
            transform_norm(image_size)

    def test_zero_image_size(self):
        image_size = 0
        with pytest.raises(ValueError, match='Image size must be a positive number'):
            transform_norm(image_size)

    def test_result_type(self):
        result_image = self.transform(self.image)
        assert isinstance(result_image, torch.Tensor)

    def test_result_shape(self):
        result_image = self.transform(self.image)
        assert result_image.shape == (3, self.image_size, self.image_size)

    def test_normalisation(self):
        mean = torch.mean(self.tensor, dim=(1, 2))
        std = torch.std(self.tensor, dim=(1, 2))
        transform = transform_norm(self.image_size, mean, std)
        images_norm = transform(self.image)
        assert np.allclose(torch.mean(images_norm, dim=(1, 2)), (0, 0, 0), atol=1e-2, rtol=0)
        assert np.allclose(torch.std(images_norm, dim=(1, 2)), (1, 1, 1), atol=1e-2, rtol=0)


class TestTransformUnnorm:
    image_size = 256
    transform = transform_unnorm(image_size)
    tensor = torch.rand(3, image_size, image_size)
    image = transforms.ToPILImage()(tensor)

    def test_negative_image_size(self):
        image_size = -5
        with pytest.raises(ValueError, match='Image size must be a positive number'):
            transform_unnorm(image_size)

    def test_zero_image_size(self):
        image_size = 0
        with pytest.raises(ValueError, match='Image size must be a positive number'):
            transform_unnorm(image_size)

    def test_result_type(self):
        result_image = self.transform(self.image)
        assert isinstance(result_image, torch.Tensor)

    def test_result_shape(self):
        result_image = self.transform(self.image)
        assert result_image.shape == (3, self.image_size, self.image_size)

    def test_normalisation_absence(self):
        transform = transform_unnorm(self.image_size)
        images_unnorm = transform(self.image)
        # torch.rand returns values from uniform distribution on range (a,b] = (0, 1]
        # mean: (b - a) / 2, std: (b - a)^2 / sqrt(12)
        assert np.allclose(torch.mean(images_unnorm, dim=(1, 2)), (0.5, 0.5, 0.5), atol=1e-2, rtol=0)
        assert np.allclose(torch.std(images_unnorm, dim=(1, 2)), (1 / 12 ** 0.5, 1 / 12 ** 0.5, 1 / 12 ** 0.5),
                           atol=1e-2, rtol=0)


class TestLoadData:
    image_size = 64
    batch_size = 9

    def test_existing_dir(self):
        dataloader, dataset = load_data(
            r'datasets/anime_partitioned/',
            self.image_size,
            self.batch_size,
        )
        assert dataloader
        assert dataset

    def test_non_existing_dir(self):
        with pytest.raises(FileNotFoundError):
            dataloader, dataset = load_data(
                r'datasets/some_dir/',
                self.image_size,
                self.batch_size,
            )

    def test_batches_count(self):
        dataloader, dataset = load_data(
            r'datasets/anime_partitioned/',
            self.image_size,
            self.batch_size,
        )
        assert math.ceil(len(dataset) / float(self.batch_size)) == len(dataloader)


class TestLoadCUB:
    image_size = 64
    batch_size = 9

    def test_existing_dir(self):
        dataloader, dataset = load_cub(
            r'datasets/',
            self.image_size,
            self.batch_size,
        )
        assert dataloader
        assert dataset

    def test_non_existing_dir(self):
        with pytest.raises(FileNotFoundError):
            dataloader, dataset = load_cub(
                r'datasets/CUB',
                self.image_size,
                self.batch_size,
            )

# TEST CUB
