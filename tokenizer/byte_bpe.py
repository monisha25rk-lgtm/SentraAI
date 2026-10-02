from pathlib import Path
from collections import Counter

corpus_path = Path("data/raw/corpus_100.txt")
merges_path = Path("tokenizer/merges.txt")

# Read the corpus and convert it to UTF-8 bytes.
text = corpus_path.read_text(encoding="utf-8")
data = list(text.encode("utf-8"))

print("Initial number of symbols:", len(data))


def apply_merge(data, pair, new_token_id):
    merged_data = []
    i = 0

    while i < len(data):
        if (
            i < len(data) - 1
            and data[i] == pair[0]
            and data[i + 1] == pair[1]
        ):
            merged_data.append(new_token_id)
            i += 2
        else:
            merged_data.append(data[i])
            i += 1

    return merged_data


# Load previously learned merges.
learned_merges = []

if merges_path.exists():
    lines = merges_path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if not line.strip():
            continue

        first, second, token_id = map(int, line.split())
        learned_merges.append((first, second, token_id))


# Apply all previously learned merges.
for first, second, token_id in learned_merges:
    data = apply_merge(data, (first, second), token_id)


if learned_merges:
    print("Previously learned merges:", len(learned_merges))
    print("Symbols after applying previous merges:", len(data))


# The next token ID comes after the 256 byte tokens
# and any previously learned tokens.
next_token_id = 256 + len(learned_merges)

new_merges = []

while True:

    pair_counts = Counter(zip(data, data[1:]))

    if not pair_counts:
        break

    most_common_pair, count = pair_counts.most_common(1)[0]

    # Stop when no pair occurs more than once.
    if count < 2:
        break

    new_token_id = next_token_id

    data = apply_merge(
        data,
        most_common_pair,
        new_token_id
    )

    new_merges.append(
        (
            most_common_pair[0],
            most_common_pair[1],
            new_token_id
        )
    )

    print(
        f"Merge #{len(learned_merges) + len(new_merges)}: "
        f"{most_common_pair} -> {new_token_id} "
        f"({count} occurrences)"
    )

    next_token_id += 1


# Append newly learned merges to the existing merge file.
if new_merges:
    with merges_path.open("a", encoding="utf-8") as f:
        for first, second, token_id in new_merges:
            f.write(f"{first} {second} {token_id}\n")


print()
print("Total learned merges:", len(learned_merges) + len(new_merges))
print("New merges learned:", len(new_merges))
print("Final number of symbols:", len(data))
print("Saved merges to:", merges_path)
def encode(text, merges):
    data = list(text.encode("utf-8"))

    for first, second, token_id in merges:
        data = apply_merge(data, (first, second), token_id)

    return data

test_sentence = "The server rejected an invalid request."

encoded = encode(test_sentence, learned_merges)

print("\nTest sentence:")
print(test_sentence)

print("\nEncoded token IDs:")
print(encoded)

print("\nNumber of tokens:")
print(len(encoded))

def build_token_vocabulary(merges):
    vocabulary = {i: bytes([i]) for i in range(256)}

    for first, second, token_id in merges:
        vocabulary[token_id] = vocabulary[first] + vocabulary[second]

    return vocabulary

def decode(token_ids, vocabulary):
    byte_sequence = b""

    for token_id in token_ids:
        byte_sequence += vocabulary[token_id]

    return byte_sequence.decode("utf-8")

vocabulary = build_token_vocabulary(learned_merges)

decoded = decode(encoded, vocabulary)

print("\nDecoded text:")
print(decoded)

print("\nRound-trip successful:")
print(decoded == test_sentence)
