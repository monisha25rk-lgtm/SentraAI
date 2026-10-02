import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            sequence_length,
            embedding_dim
        )

    def forward(self, positions):
        return self.position_embedding(positions)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        embedding_dim
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            sequence_length,
            embedding_dim
        )

    def forward(self, token_ids):

        token_vectors = self.token_embedding(
            token_ids
        )

        sequence_length = token_ids.size(1)

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        return token_vectors + position_vectors