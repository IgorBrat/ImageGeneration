import pytest
import torch
import numpy as np

from ml.utils.tensor_logic import (get_uniform_noise, get_normal_noise, get_one_hot_labels, reparameterise, matrix_sqrt,
                                   concat_vectors, gram_matrix)


class TestUniformNoise:
    cpu = 'cpu'
    cuda = 'cuda'
    shape1 = (3, 3, 128, 128)
    shape2 = (1, 32, 32)
    shape3 = tuple([100])

    def test_shape(self):
        assert get_uniform_noise(self.shape1, self.cpu).shape == self.shape1
        assert get_uniform_noise(self.shape2, self.cpu).shape == self.shape2
        assert get_uniform_noise(self.shape3, self.cpu).shape == self.shape3

    def test_distribution(self):
        noise1 = get_uniform_noise(self.shape1, self.cpu)
        noise2 = get_uniform_noise(self.shape2, self.cpu)
        noise3 = get_uniform_noise(self.shape3, self.cpu)
        assert np.isclose(torch.mean(noise1), 0.5, atol=1e-2, rtol=0)
        assert np.isclose(torch.std(noise1), 1 / 12 ** 0.5, atol=1e-2, rtol=0)
        assert np.isclose(torch.mean(noise2), 0.5, atol=1e-1, rtol=0)
        assert np.isclose(torch.std(noise2), 1 / 12 ** 0.5, atol=1e-1, rtol=0)
        assert np.isclose(torch.mean(noise3), 0.5, atol=5e-1, rtol=0)
        assert np.isclose(torch.std(noise3), 1 / 12 ** 0.5, atol=5e-1, rtol=0)

    def test_device(self):
        assert get_uniform_noise(self.shape1, self.cpu).get_device() == -1
        assert get_uniform_noise(self.shape2, self.cpu).get_device() == -1
        if torch.cuda.is_available():
            assert get_uniform_noise(self.shape3, self.cuda).get_device() > 0


class TestNormalNoise:
    cpu = 'cpu'
    cuda = 'cuda'
    shape1 = (3, 3, 128, 128)
    shape2 = (1, 32, 32)
    shape3 = tuple([100])

    def test_shape(self):
        assert get_uniform_noise(self.shape1, self.cpu).shape == self.shape1
        assert get_uniform_noise(self.shape2, self.cpu).shape == self.shape2
        assert get_uniform_noise(self.shape3, self.cpu).shape == self.shape3

    def test_distribution(self):
        noise1 = get_normal_noise(self.shape1, self.cpu)
        noise2 = get_normal_noise(self.shape2, self.cpu)
        noise3 = get_normal_noise(self.shape3, self.cpu)
        assert np.isclose(torch.mean(noise1), 0, atol=1e-2, rtol=0)
        assert np.isclose(torch.std(noise1), 1, atol=1e-2, rtol=0)
        assert np.isclose(torch.mean(noise2), 0, atol=1e-1, rtol=0)
        assert np.isclose(torch.std(noise2), 1, atol=1e-1, rtol=0)
        assert np.isclose(torch.mean(noise3), 0, atol=5e-1, rtol=0)
        assert np.isclose(torch.std(noise3), 1, atol=5e-1, rtol=0)

    def test_device(self):
        assert get_uniform_noise(self.shape1, self.cpu).get_device() == -1
        assert get_uniform_noise(self.shape2, self.cpu).get_device() == -1
        if torch.cuda.is_available():
            assert get_uniform_noise(self.shape3, self.cuda).get_device() > 0


class TestOneHot:
    def test_excess_labels(self):
        labels = torch.Tensor([1, 5, 2, 10]).long()
        classes = 9
        with pytest.raises(RuntimeError, match='Class values must be smaller than num_classes.'):
            one_hot = get_one_hot_labels(labels, classes)
        labels = torch.Tensor(5, 3, 128, 128).random_(1, 20).long()
        classes = 15
        with pytest.raises(RuntimeError, match='Class values must be smaller than num_classes.'):
            one_hot = get_one_hot_labels(labels, classes)

    def test_incorrect_type(self):
        labels = torch.Tensor([1, 5, 2, 10])
        classes = 11
        with pytest.raises(RuntimeError, match='one_hot is only applicable to index tensor.'):
            one_hot = get_one_hot_labels(labels, classes)
        labels = 10
        with pytest.raises(TypeError, match='must be Tensor'):
            one_hot = get_one_hot_labels(labels, classes)

    def test_correct_labels(self):
        labels = torch.Tensor([1, 4, 2, 1]).long()
        classes = 5
        assert torch.equal(get_one_hot_labels(labels, classes),
                           torch.Tensor([
                               [0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1],
                               [0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0],
                           ]))
        labels = torch.tensor([9, 5, 3, 2, 8, 2, 9, 0])
        classes = 10
        assert torch.equal(get_one_hot_labels(labels, classes),
                           torch.Tensor([
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ]))


class TestReparameterise:
    def test_zero(self):
        mean = torch.Tensor(0)
        std = torch.Tensor(0)
        assert torch.equal(reparameterise(mean, std), torch.Tensor(0))

    def test_reparameterise(self):
        mean = torch.Tensor([3])
        std = torch.Tensor([2])
        reparameterised = reparameterise(mean, std)
        assert np.isclose(torch.mean(reparameterised).detach().numpy(), 8.43, atol=1e-1, rtol=0)
        mean = torch.Tensor([10])
        std = torch.Tensor([8])
        reparameterised = reparameterise(mean, std)
        assert np.isclose(torch.mean(reparameterised).detach().numpy(), 446.79, atol=1e-1, rtol=0)


class TestMatrixSqrt:
    def test_empty(self):
        mat = torch.Tensor()
        with pytest.raises(ValueError, match='Non-matrix input to matrix function.'):
            res = matrix_sqrt(mat)

    def test_1d(self):
        mat = torch.Tensor([0, 10, 9, 2])
        with pytest.raises(ValueError, match='Non-matrix input to matrix function.'):
            res = matrix_sqrt(mat)

    def test_2d(self):
        mat1 = torch.Tensor([[1, 5, 6], [1, 5, 6], [1, 5, 6], ])
        mat1_sqr = matrix_sqrt(mat1)
        assert torch.equal(torch.round(torch.matmul(mat1_sqr, mat1_sqr)).int(), mat1)
        mat2 = torch.Tensor([[1, 5, 6, 10], [2, 3, 4, 1], [0, 11, 2, 5], [1, 2, 1, 0], ])
        mat2_sqr = matrix_sqrt(mat2)
        # explicit .int() without rounding first acts weird when rounding itself
        assert torch.equal(torch.round(torch.matmul(mat2_sqr, mat2_sqr)).int(), mat2)


class TestConcat:
    def test_empty(self):
        assert torch.equal(concat_vectors(torch.Tensor(), torch.Tensor()), torch.Tensor())

    def test_dimension(self):
        with pytest.raises(IndexError):
            concat_vectors(torch.Tensor([1, 3]), torch.Tensor([2]), dim=3)
        with pytest.raises(RuntimeError, match='Tensors must have same number of dimensions'):
            concat_vectors(torch.Tensor([[1, 3], [2, 4]]), torch.Tensor([2]), dim=0)
        assert concat_vectors(torch.Tensor([1, 3]), torch.Tensor([2]), dim=0).shape == tuple([3])
        assert concat_vectors(torch.Tensor([[1, 3], [2, 4]]), torch.Tensor([[2, 10]]), dim=0).shape == (3, 2)
        assert concat_vectors(torch.Tensor([[1, 3], [2, 4]]), torch.Tensor([[2, 4], [2, 3]]), dim=1).shape == (2, 4)

    def test_values(self):
        assert torch.equal(
            concat_vectors(torch.Tensor([[1, 3], [2, 4]]), torch.Tensor([[2, 10]]), dim=0),
            torch.Tensor([[1, 3], [2, 4], [2, 10]])
        )
        assert torch.equal(
            concat_vectors(torch.Tensor([[1, 3], [2, 4]]), torch.Tensor([[2, 10], [-5, 0]]), dim=1),
            torch.Tensor([[1, 3, 2, 10], [2, 4, -5, 0]])
        )


class TestGramMatrix:
    def test_incorrect(self):
        with pytest.raises(ValueError, match='Parse non-empty matrix'):
            gram_matrix(torch.Tensor([]))
        with pytest.raises(TypeError, match='Matrix must be 2-dimensional'):
            gram_matrix(torch.Tensor([1, 2]))

    def test_correct(self):
        assert torch.equal(
            gram_matrix(torch.Tensor([[2, 4], [3, 5]])),
            torch.Tensor([[20, 26], [26, 34]])
        )
        assert torch.equal(
            gram_matrix(torch.Tensor([[2, 4, 1], [3, 5, 2], [0, -2, 3]])),
            torch.Tensor([[21, 28, -5], [28, 38, -4], [-5, -4, 13]])
        )

