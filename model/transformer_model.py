import torch.nn as nn

from .model import InputEmbedding
from .transformer_block import TransformerBlock


class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        sequence_length,
        embedding_dim,
        num_heads,
        num_layers
    ):
        super().__init__()

        # --------------------------------
        # INPUT EMBEDDING
        # --------------------------------

        self.embedding = InputEmbedding(
            vocab_size,
            sequence_length,
            embedding_dim
        )

        # --------------------------------
        # TRANSFORMER BLOCKS
        # --------------------------------

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_number=i + 1
                )
                for i in range(num_layers)
            ]
        )

        # --------------------------------
        # LANGUAGE MODEL HEAD
        # --------------------------------

        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):

        # --------------------------------
        # 1. TOKEN + POSITION EMBEDDING
        # --------------------------------

        x = self.embedding(token_ids)

        # --------------------------------
        # 2. TRANSFORMER BLOCKS
        # --------------------------------

        for block in self.transformer_blocks:
            x = block(x)

        # --------------------------------
        # 3. VOCABULARY PROJECTION
        # --------------------------------

        logits = self.output_projection(x)

        return logits