import os
import re
import random
import json
import collections

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
random.seed(42)

nltk.download("punkt", quiet=True)

DATA_PATH = os.path.join("data", "imdb_reviews.csv")
VOCAB_PATH = os.path.join("data", "vocab.json")
STATS_PATH = os.path.join("data", "dataset_stats.json")

MAX_VOCAB_SIZE = 10_000
SEQ_LENGTHS = [25, 50, 100]


def basic_clean(text: str) -> str:
    """
    Lowercase, remove punctuation/special chars, collapse spaces.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # keep letters, numbers, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """
    Simple whitespace tokenizer after cleaning.
    (Assignment allows nltk.word_tokenize or similar.)
    """
    return text.split()


def build_vocab(texts, max_words: int):
    """
    Build vocabulary of most frequent words from training texts.
    Indexing:
      0 -> PAD
      1..max_words -> most frequent words
    """
    counter = collections.Counter()
    for txt in texts:
        counter.update(tokenize(txt))

    most_common = counter.most_common(max_words)
    word_index = {word: idx for idx, (word, _) in enumerate(most_common, start=1)}

    return word_index, counter


def texts_to_sequences(texts, word_index):
    """
    Convert list of texts to list of sequences of word IDs.
    Tokens not in vocab are dropped (since we 'keep top 10k words').
    """
    sequences = []
    for txt in texts:
        tokens = tokenize(txt)
        seq = [word_index[t] for t in tokens if t in word_index]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences, maxlen: int):
    """
    Left-truncate, right-pad with 0 (PAD) to fixed length maxlen.
    """
    padded = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) >= maxlen:
            padded[i] = np.array(seq[-maxlen:])  # take last maxlen tokens
        else:
            padded[i, : len(seq)] = np.array(seq)
    return padded


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print("\nColumns:", df.columns.tolist())
    print("\nLabel distribution:")
    print(df["sentiment"].value_counts())

    # Clean text
    print("\nApplying basic cleaning ...")
    df["clean_review"] = df["review"].apply(basic_clean)
    df["length"] = df["clean_review"].str.split().str.len()

    print("\nReview length stats (tokens after cleaning):")
    print(df["length"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    # Encode labels: positive -> 1, negative -> 0
    label_map = {"positive": 1, "negative": 0}
    df["label"] = df["sentiment"].map(label_map).astype(int)

    # 50/50 split (25k / 25k), stratified and reproducible
    print("\nSplitting into train/test (50/50, stratified) ...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        stratify=df["label"],
        random_state=42,
        shuffle=True,
    )

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Build vocab from TRAIN ONLY
    print("\nBuilding vocabulary from training data ...")
    word_index, freq_counter = build_vocab(train_df["clean_review"].tolist(), MAX_VOCAB_SIZE)
    vocab_size = len(word_index) + 1  # +1 for PAD=0

    print(f"Vocab size (including PAD=0): {vocab_size}")

    # Save vocab to JSON
    with open(VOCAB_PATH, "w") as f:
        json.dump({"word_index": word_index}, f)
    print(f"Saved vocab to {VOCAB_PATH}")

    # Convert texts to sequences
    print("\nConverting texts to sequences ...")
    train_seqs = texts_to_sequences(train_df["clean_review"].tolist(), word_index)
    test_seqs = texts_to_sequences(test_df["clean_review"].tolist(), word_index)

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    # Save stats for the report
    stats = {
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "vocab_size_including_pad": int(vocab_size),
        "avg_length": float(df["length"].mean()),
        "median_length": float(df["length"].median()),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved dataset stats to {STATS_PATH}")

    # Create and save padded datasets for each sequence length
    for seq_len in SEQ_LENGTHS:
        print(f"\nPadding sequences to length {seq_len} ...")
        x_train = pad_sequences(train_seqs, seq_len)
        x_test = pad_sequences(test_seqs, seq_len)

        out_path = os.path.join("data", f"imdb_seq{seq_len}.npz")
        np.savez_compressed(
            out_path,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        print(
            f"Saved preprocessed data for seq_len={seq_len} to {out_path} "
            f"with shapes x_train={x_train.shape}, x_test={x_test.shape}"
        )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
