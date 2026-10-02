import torch.nn as nn

from attention import SelfAttention
from normalization import ResidualLayerNorm
from feed_forward import FeedForward


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

        print(
            f"\n========== TRANSFORMER BLOCK "
            f"{self.block_number} =========="
        )

        # --------------------------------
        # 1. INPUT
        # --------------------------------

        print("\n[1] INPUT")
        print("Shape:", x.shape)
        print("First 5 values:", x[0, 0, :5])

        # --------------------------------
        # 2. SELF-ATTENTION
        # --------------------------------

        attention_output = self.attention(x)

        print("\n[2] SELF-ATTENTION OUTPUT")
        print("Shape:", attention_output.shape)
        print(
            "First 5 values:",
            attention_output[0, 0, :5]
        )

        # --------------------------------
        # 3. RESIDUAL ADDITION
        # --------------------------------

        residual_attention = (
            x + attention_output
        )

        print("\n[3] RESIDUAL ADDITION")
        print("Shape:", residual_attention.shape)
        print(
            "First 5 values:",
            residual_attention[0, 0, :5]
        )

        # --------------------------------
        # 4. LAYER NORMALIZATION
        # --------------------------------

        x = self.attention_norm.layer_norm(
            residual_attention
        )

        print("\n[4] AFTER LAYER NORMALIZATION")
        print("Shape:", x.shape)
        print(
            "First 5 values:",
            x[0, 0, :5]
        )
        print(
            "Mean:",
            x.mean().item()
        )
        print(
            "Std:",
            x.std().item()
        )

        # --------------------------------
        # 5. FEED-FORWARD NETWORK
        # --------------------------------

        feed_forward_output = (
            self.feed_forward(x)
        )

        print("\n[5] FEED-FORWARD OUTPUT")
        print("Shape:", feed_forward_output.shape)
        print(
            "First 5 values:",
            feed_forward_output[0, 0, :5]
        )

        # --------------------------------
        # 6. SECOND RESIDUAL ADDITION
        # --------------------------------

        residual_ffn = (
            x + feed_forward_output
        )

        print("\n[6] SECOND RESIDUAL ADDITION")
        print("Shape:", residual_ffn.shape)
        print(
            "First 5 values:",
            residual_ffn[0, 0, :5]
        )

        # --------------------------------
        # 7. SECOND LAYER NORMALIZATION
        # --------------------------------

        x = self.feed_forward_norm.layer_norm(
            residual_ffn
        )

        print("\n[7] FINAL TRANSFORMER OUTPUT")
        print("Shape:", x.shape)
        print(
            "First 5 values:",
            x[0, 0, :5]
        )
        print(
            "Mean:",
            x.mean().item()
        )
        print(
            "Std:",
            x.std().item()
        )

        print(
            f"\n========== END TRANSFORMER BLOCK "
            f"{self.block_number} ==========\n"
        )

        return x