
import sys
from pathlib import Path

import torch
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
# IMPORTS
# ========================================

from data_pipeline import load_training_data
from model.transformer_model import TransformerModel
from training.loss import LanguageModelLoss


# ========================================
# MODEL VERSION
# ========================================

MODEL_VERSION = 1

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "model_v1"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "checkpoint.pt"
)


# ========================================
# LOAD TRAINING CONFIGURATION
# ========================================

TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "training_config.yaml"
)

with open(
    TRAINING_CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    training_config = yaml.safe_load(file)


# ========================================
# READ TRAINING PARAMETERS
# ========================================

batch_size = training_config[
    "training"
][
    "batch_size"
]

epochs = training_config[
    "training"
][
    "epochs"
]

learning_rate = training_config[
    "training"
][
    "learning_rate"
]


# ========================================
# LOAD TRAINING DATA
# ========================================

(
    inputs,
    targets,
    dataset_config,
    model_config,
    tokenizer_config,
    vocabulary_size
) = load_training_data()


# ========================================
# CONVERT TOKEN IDs TO TENSORS
# ========================================

inputs = torch.tensor(
    inputs,
    dtype=torch.long
)

targets = torch.tensor(
    targets,
    dtype=torch.long
)


# ========================================
# CREATE FOUNDATION MODEL
# ========================================

model = TransformerModel(
    vocab_size=vocabulary_size,
    sequence_length=model_config["sequence_length"],
    embedding_dim=model_config["embedding_dim"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"]
)


# ========================================
# CREATE OPTIMIZER
# ========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ========================================
# CREATE LOSS FUNCTION
# ========================================

loss_function = LanguageModelLoss()


# ========================================
# TRAINING INFORMATION
# ========================================

number_of_samples = inputs.size(0)

number_of_batches = (
    number_of_samples
    + batch_size
    - 1
) // batch_size


# ========================================
# DISPLAY TRAINING CONFIGURATION
# ========================================

print(
    "========================================"
)

print(
    "SENTRA FOUNDATION MODEL TRAINING"
)

print(
    "========================================"
)

print(
    "Model version:",
    MODEL_VERSION
)

print(
    "Dataset version:",
    dataset_config["dataset_version"]
)

print(
    "Tokenizer version:",
    tokenizer_config["tokenizer_version"]
)

print()

print(
    "Training samples:",
    number_of_samples
)

print(
    "Batch size:",
    batch_size
)

print(
    "Batches per epoch:",
    number_of_batches
)

print(
    "Epochs:",
    epochs
)

print(
    "Learning rate:",
    learning_rate
)

print()


# ========================================
# TRAINING LOOP
# ========================================

for epoch in range(
    epochs
):

    epoch_loss = 0.0

    # --------------------------------
    # BATCH LOOP
    # --------------------------------

    for batch_start in range(
        0,
        number_of_samples,
        batch_size
    ):

        batch_end = (
            batch_start
            + batch_size
        )

        batch_inputs = inputs[
            batch_start:batch_end
        ]

        batch_targets = targets[
            batch_start:batch_end
        ]

        # ----------------------------
        # FORWARD PASS
        # ----------------------------

        logits = model(
            batch_inputs
        )

        # ----------------------------
        # CALCULATE LOSS
        # ----------------------------

        loss = loss_function(
            logits,
            batch_targets
        )

        # ----------------------------
        # CLEAR OLD GRADIENTS
        # ----------------------------

        optimizer.zero_grad()

        # ----------------------------
        # BACKPROPAGATION
        # ----------------------------

        loss.backward()

        # ----------------------------
        # UPDATE MODEL WEIGHTS
        # ----------------------------

        optimizer.step()

        # ----------------------------
        # ACCUMULATE LOSS
        # ----------------------------

        epoch_loss += loss.item()

    # --------------------------------
    # AVERAGE EPOCH LOSS
    # --------------------------------

    average_epoch_loss = (
        epoch_loss
        / number_of_batches
    )

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"| Loss: {average_epoch_loss:.6f}"
    )


# ========================================
# TRAINING COMPLETED
# ========================================

print()

print(
    "========================================"
)

print(
    "TRAINING COMPLETED"
)

print(
    "========================================"
)

print(
    "Final epoch loss:",
    average_epoch_loss
)

print()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),

    "epoch": epochs,
    "model_version": MODEL_VERSION,
    "dataset_version": dataset_config["dataset_version"],
    "tokenizer_version": tokenizer_config["tokenizer_version"],

    "model_config": model_config,
    "training_config": training_config,
}

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

print(
    "Model V1 training run completed."
)

print(
    "Checkpoint saved to:",
    CHECKPOINT_PATH
)





