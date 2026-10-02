from pathlib import Path
import yaml
import sys

sys.path.append("tokenizer")

from byte_bpe import encode


def load_dataset_config(config_path):
    """
    Load the dataset configuration from YAML.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_text_dataset(config):
    """
    Load a text dataset using the path specified
    in the dataset configuration.
    """

    dataset_path = Path(config["source"]["path"])

    with open(dataset_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def load_tokenizer():
    """
    Load the persisted Byte-Level BPE merges.
    """

    merges = []

    with open("tokenizer/merges.txt", "r", encoding="utf-8") as file:
        for line in file:
            first, second, token_id = map(int, line.split())
            merges.append((first, second, token_id))

    return merges
def build_token_stream(text, merges):
    """
    Convert raw text into a continuous stream of token IDs.
    """

    token_ids = encode(text, merges)

    return token_ids

def create_sequences(token_stream, sequence_length):
    """
    Split the token stream into fixed-length sequences.
    """

    sequences = []

    for start in range(
        0,
        len(token_stream) - sequence_length + 1,
        sequence_length
    ):
        sequence = token_stream[
            start:start + sequence_length
        ]

        sequences.append(sequence)

    return sequences

def create_input_target_pairs(sequences):
    """
    Create input and target sequences for next-token prediction.
    """

    inputs = []
    targets = []

    for sequence in sequences:
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]

        inputs.append(input_sequence)
        targets.append(target_sequence)

    return inputs, targets



if __name__ == "__main__":

    config = load_dataset_config(
        "configs/dataset_config.yaml"
    )

    text = load_text_dataset(config)

    merges = load_tokenizer()

    token_stream = build_token_stream(text, merges)
    sequence_length = 4

    sequences = create_sequences(
    token_stream,
    sequence_length
     )
    inputs, targets = create_input_target_pairs(sequences)

    print("First input:", inputs[0])
    print("First target:", targets[0])

    print("Sequence length:", sequence_length)
    print("Number of sequences:", len(sequences))
    print("First sequence:", sequences[0])

    print("Dataset version:", config["dataset_version"])
    print("Tokenizer version:", config["tokenizer_version"])
    print("Dataset characters:", len(text))
    print("Number of merges:", len(merges))
    print("Number of token IDs:", len(token_stream))
    print("First 20 token IDs:", token_stream[:20])