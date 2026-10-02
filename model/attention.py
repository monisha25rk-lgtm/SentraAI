import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_output = attention_weights @ V

        attention_output = attention_output.transpose(
            1, 2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        return self.output(attention_output)