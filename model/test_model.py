import torch

from model import InputEmbedding


vocab_size = 687
sequence_length = 4
embedding_dim = 128
batch_size = 2


embedding_layer = InputEmbedding(
    vocab_size,
    sequence_length,
    embedding_dim
)


token_ids = torch.tensor([
    [260, 342, 495, 453],
    [342, 495, 453, 302]
])


output = embedding_layer(token_ids)


print("Input shape:", token_ids.shape)
print("Output shape:", output.shape)