from sentence_transformers import SentenceTransformer
import torch
from ml.utils.tensor_logic import reparameterise


def preprocess_attributes(attributes: tuple[str]):
    """
    Preprocess CUB attributes to readable form
    :param attributes: given attributes
    :return: preprocessed attributes
    """
    attributes = list(attributes)
    for idx in range(len(attributes)):
        curr_attributes = attributes[idx]
        curr_attributes = curr_attributes.replace('::', ' ').replace('has', '').replace('_', ' ').replace('  ', ' ')
        attributes[idx] = curr_attributes
    return attributes


class TextEncoder:
    """
    Text Encoder model for embedding given text queries
    """

    def __init__(self, model: str):
        self.model_name = model
        self.model = SentenceTransformer(model)

    def get_embedding_dimension(self):
        """
        Get embedding dimension of model
        """
        return self.model.get_sentence_embedding_dimension()

    def encode(self, text):
        """
        Encode whole text
        :param text: given text
        :return: text embedding of torch.Tensor
        """
        return self.model.encode(text, convert_to_tensor=True)


class ConditionAugmentation(torch.nn.Module):
    """
    Condition Augmentation used to learn fine representation of given text embeddings in Text2Image models
    """

    def __init__(self, to_reparameterise: bool = True, text_embedding_dim: int = 300, text_embedding_latent: int = 128):
        super(ConditionAugmentation, self).__init__()
        self.to_reparameterise = to_reparameterise
        self.embed_mean = torch.nn.Linear(text_embedding_latent, text_embedding_latent)
        self.embed_log_variance = torch.nn.Linear(text_embedding_latent, text_embedding_latent)
        self.text_embed = torch.nn.Sequential(
            torch.nn.Linear(text_embedding_dim, text_embedding_latent),
            torch.nn.BatchNorm1d(text_embedding_latent),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, text_emb: torch.Tensor):
        text_emb_low_dim = self.text_embed(text_emb)
        if not self.to_reparameterise:
            return text_emb_low_dim
        mean = self.embed_mean(text_emb_low_dim)
        log_variance = self.embed_log_variance(text_emb_low_dim)
        return reparameterise(mean, log_variance), mean, log_variance
