import torch.nn as nn

from .attention import SelfAttention
from .normalization import ResidualLayerNorm
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_number
    ):
        super().__init__()

        self.block_number = block_number

        self.attention = SelfAttention(
            embedding_dim,
            num_heads
        )

        self.attention_norm = ResidualLayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )

        self.feed_forward_norm = ResidualLayerNorm(
            embedding_dim
        )

    def forward(self, x):

        attention_output = self.attention(x)

        x = self.attention_norm(
            x,
            attention_output
        )

        feed_forward_output = self.feed_forward(x)

        x = self.feed_forward_norm(
            x,
            feed_forward_output
        )

        return x