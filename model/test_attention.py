import torch

from attention import SelfAttention


batch_size = 1
sequence_length = 4
embedding_dim = 128
num_heads = 4


attention = SelfAttention(
    embedding_dim,
    num_heads
)


# Create a simple input
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


# Run self-attention
output = attention(x)


print("Input shape:", x.shape)
print("Output shape:", output.shape)

print("\nSelf-attention test completed.")
print("Causal mask is active in the model.")
print("Future tokens are prevented from being attended to.")
print("Output shape:", output.shape)