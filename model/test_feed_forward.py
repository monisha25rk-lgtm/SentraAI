import torch

from feed_forward import FeedForward


batch_size = 1
sequence_length = 4
embedding_dim = 128


feed_forward = FeedForward(
    embedding_dim
)

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

output = feed_forward(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)