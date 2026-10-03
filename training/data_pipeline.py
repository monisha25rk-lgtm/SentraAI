
from pathlib import Path
import sys
import yaml


# ========================================
# PROJECT ROOT
# ========================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ========================================
# IMPORT TOKENIZER
# ========================================

from tokenizer.byte_bpe import (
    encode,
    load_tokenizer
)


# ========================================
# LOAD DATASET CONFIGURATION
# ========================================

def load_dataset_config():

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "dataset_config.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    return config


# ========================================
# LOAD MODEL CONFIGURATION
# ========================================
#
# Sequence length belongs to the model
# architecture, so it is loaded from
# model_config.yaml.
#
# ========================================

def load_model_config():

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "model_config.yaml"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    return config


# ========================================
# LOAD TEXT DATASET
# ========================================

def load_text_dataset(
    config
):

    dataset_path = (
        PROJECT_ROOT
        / config["source"]["path"]
    )

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return text


# ========================================
# BUILD TOKEN STREAM
# ========================================

def build_token_stream(
    text,
    merges
):

    token_ids = encode(
        text,
        merges
    )

    return token_ids


# ========================================
# CREATE FIXED-LENGTH SEQUENCES
# ========================================

def create_sequences(
    token_stream,
    sequence_length
):

    sequences = []

    for start in range(
        0,
        len(token_stream) - sequence_length + 1,
        sequence_length
    ):

        sequence = token_stream[
            start:start + sequence_length
        ]

        sequences.append(
            sequence
        )

    return sequences


# ========================================
# CREATE INPUT / TARGET PAIRS
# ========================================

def create_input_target_pairs(
    sequences
):

    inputs = []

    targets = []

    for sequence in sequences:

        input_sequence = sequence[:-1]

        target_sequence = sequence[1:]

        inputs.append(
            input_sequence
        )

        targets.append(
            target_sequence
        )

    return (
        inputs,
        targets
    )


# ========================================
# LOAD COMPLETE TRAINING DATA
# ========================================

def load_training_data():

    # --------------------------------
    # DATASET CONFIG
    # --------------------------------

    dataset_config = (
        load_dataset_config()
    )

    # --------------------------------
    # MODEL CONFIG
    # --------------------------------

    model_config = (
        load_model_config()
    )

    # --------------------------------
    # LOAD DATASET
    # --------------------------------

    text = load_text_dataset(
        dataset_config
    )

    # --------------------------------
    # LOAD FIXED TOKENIZER
    # --------------------------------

    (
        merges,
        vocabulary,
        vocabulary_size,
        tokenizer_config
    ) = load_tokenizer()

    # --------------------------------
    # TOKENIZE DATASET
    # --------------------------------

    token_stream = build_token_stream(
        text,
        merges
    )

    # --------------------------------
    # GET MODEL SEQUENCE LENGTH
    # --------------------------------

    sequence_length = (
        model_config[
            "sequence_length"
        ]
    )

    # --------------------------------
    # CREATE SEQUENCES
    # --------------------------------

    sequences = create_sequences(
        token_stream,
        sequence_length
    )

    # --------------------------------
    # CREATE INPUT / TARGET PAIRS
    # --------------------------------

    inputs, targets = (
        create_input_target_pairs(
            sequences
        )
    )

    # --------------------------------
    # RETURN ALL REQUIRED INFORMATION
    # --------------------------------

    return (
        inputs,
        targets,
        dataset_config,
        model_config,
        tokenizer_config,
        vocabulary_size
    )

