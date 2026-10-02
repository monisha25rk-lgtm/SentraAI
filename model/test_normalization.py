import torch

from normalization import ResidualLayerNorm


embedding_dim = 128
batch_size = 1
sequence_length = 4


normalization = ResidualLayerNorm(
    embedding_dim
)

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

sublayer_output = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

output = normalization(
    x,
    sublayer_output
)

print("Input shape:", x.shape)
print("Sublayer output shape:", sublayer_output.shape)
print("Final output shape:", output.shape)