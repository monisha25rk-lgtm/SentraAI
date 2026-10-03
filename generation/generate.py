import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from tokenizer.byte_bpe import (
    encode,
    decode,
    load_tokenizer
)

from model.transformer_model import TransformerModel


MODEL_VERSION = 1

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "model_v1"
    / "checkpoint.pt"
)


def load_checkpoint():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    return checkpoint


def build_model(checkpoint):
    model_config = checkpoint["model_config"]

    (
        merges,
        vocabulary,
        vocabulary_size,
        tokenizer_config
    ) = load_tokenizer()

    model = TransformerModel(
        vocab_size=vocabulary_size,
        sequence_length=model_config["sequence_length"],
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        merges,
        vocabulary,
        vocabulary_size,
        model_config
    )


def generate_text(
    model,
    prompt,
    merges,
    vocabulary,
    sequence_length,
    max_new_tokens=20
):
    token_ids = encode(
        prompt,
        merges
    )

    generated_tokens = list(
        token_ids
    )

    for _ in range(
        max_new_tokens
    ):

        context = generated_tokens[
            -sequence_length:
        ]

        input_tensor = torch.tensor(
            [context],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(
                input_tensor
            )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1
        ).item()

        generated_tokens.append(
            next_token_id
        )

    return decode(
        generated_tokens,
        vocabulary
    )


def main():

    checkpoint = load_checkpoint()

    (
        model,
        merges,
        vocabulary,
        vocabulary_size,
        model_config
    ) = build_model(
        checkpoint
    )

    print(
        "========================================"
    )
    print(
        "SENTRA FOUNDATION MODEL - GENERATION"
    )
    print(
        "========================================"
    )
    print(
        "Model version:",
        checkpoint["model_version"]
    )
    print(
        "Tokenizer version:",
        checkpoint["tokenizer_version"]
    )
    print(
        "Vocabulary size:",
        vocabulary_size
    )
    print()

    prompt = input(
        "Enter prompt: "
    )

    generated_text = generate_text(
        model=model,
        prompt=prompt,
        merges=merges,
        vocabulary=vocabulary,
        sequence_length=model_config[
            "sequence_length"
        ],
        max_new_tokens=20
    )

    print()
    print(
        "Generated text:"
    )
    print(
        generated_text
    )


if __name__ == "__main__":
    main()