import torch.nn as nn


class ResidualLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x, sublayer_output):
        return self.layer_norm(
            x + sublayer_output
        )