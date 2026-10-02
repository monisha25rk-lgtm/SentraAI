import torch

from transformer_model import TransformerModel


# =================================
# MODEL CONFIGURATION
# =================================

vocab_size = 687
sequence_length = 4
embedding_dim = 128
num_heads = 4
num_layers = 4


# =================================
# CREATE COMPLETE TRANSFORMER
# =================================

model = TransformerModel(
    vocab_size=vocab_size,
    sequence_length=sequence_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers
)


# =================================
# INPUT TOKEN IDs
# =================================

token_ids = torch.tensor([
    [260, 342, 495, 453]
])


print("\n========================================")
print("FULL TRANSFORMER MODEL")
print("========================================")

print("\nInput token IDs:")
print(token_ids)

print("\nInput shape:")
print(token_ids.shape)


# =================================
# RUN COMPLETE MODEL
# =================================

output = model(token_ids)


# =================================
# FINAL OUTPUT
# =================================

print("\n========================================")
print("FINAL TRANSFORMER OUTPUT")
print("========================================")

print("Output shape:", output.shape)

print(
    "First 5 values:",
    output[0, 0, :5]
)

print("\nComplete Transformer processing finished.")