import sentence_transformers
import torch
import pytest

from ml.utils.text_processing import preprocess_attributes, TextEncoder, ConditionAugmentation


class TestPreprocessAttributes:
    def test_empty(self):
        assert not preprocess_attributes(tuple(''))

    def test_attributes_processing(self):
        assert preprocess_attributes(
            ('bird has_color::red, has_fur::blue',
             'cat has_ears::pointy, has_claws::sharp')
        ) == ['bird color red, fur blue',
              'cat ears pointy, claws sharp']
        assert preprocess_attributes(
            ('bird with red beak',
             'noisy cat',
             'dog has_ears::big, has_nose::pointy')
        ) == ['bird with red beak',
              'noisy cat',
              'dog ears big, nose pointy']


class TestTextEncoder:
    model_name = "msmarco-distilbert-base-tas-b"
    embedding_dimension = 768
    encoder = TextEncoder("msmarco-distilbert-base-tas-b")

    def test_model_attributes(self):
        assert self.encoder.model_name == self.model_name
        assert self.encoder.get_embedding_dimension() == self.embedding_dimension
        assert isinstance(self.encoder.model, sentence_transformers.SentenceTransformer)
        assert self.encoder.model.device.type == 'cpu'

    def test_encode(self):
        text = "It's a really pretty day"
        assert self.encoder.encode(text).shape == (self.embedding_dimension,)
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
                "tempor incididunt ut labore et dolore magna aliqua.")
        assert self.encoder.encode(text).shape == (self.embedding_dimension,)


class TestConditionAugmentation:
    text_embedding_dim = 400
    text_embedding_latent_dim = 200
    ca_model = ConditionAugmentation(text_embedding_dim=text_embedding_dim,
                                     text_embedding_latent=text_embedding_latent_dim)
    text_embedding_dim_2 = 400
    text_embedding_latent_dim_2 = 200
    ca_model2 = ConditionAugmentation(text_embedding_dim=text_embedding_dim_2,
                                      text_embedding_latent=text_embedding_latent_dim_2)

    def test_types(self):
        assert isinstance(self.ca_model.embed_mean, torch.nn.Linear)
        assert isinstance(self.ca_model.embed_log_variance, torch.nn.Linear)
        assert isinstance(self.ca_model.text_embed[0], torch.nn.Linear)
        assert isinstance(self.ca_model.text_embed[1], torch.nn.BatchNorm1d)
        assert isinstance(self.ca_model.text_embed[2], torch.nn.LeakyReLU)

    def test_dimension(self):
        embeddings = torch.rand(10, self.text_embedding_dim)
        latent, mean, std = self.ca_model(embeddings)
        assert latent.shape == (10, self.text_embedding_latent_dim)
        assert mean.shape == (10, self.text_embedding_latent_dim)
        assert std.shape == (10, self.text_embedding_latent_dim)
        embeddings = torch.rand(20, self.text_embedding_dim_2)
        latent, mean, std = self.ca_model2(embeddings)
        assert latent.shape == (20, self.text_embedding_latent_dim_2)
        assert mean.shape == (20, self.text_embedding_latent_dim_2)
        assert std.shape == (20, self.text_embedding_latent_dim_2)
