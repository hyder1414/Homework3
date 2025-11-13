import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class IMDBDataset(Dataset):
    """
    Simple Dataset wrapper around our preprocessed NumPy arrays.
    Each item is:
        x: LongTensor of shape (seq_len,)
        y: LongTensor scalar {0,1}
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.shape[0] == y.shape[0], "x and y must have same number of samples"
        self.x = torch.from_numpy(x).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def load_imdb_data(seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed IMDb arrays for a given sequence length.
    Returns:
        x_train, y_train, x_test, y_test (all NumPy arrays)
    """
    path = os.path.join("data", f"imdb_seq{seq_len}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed file not found: {path}")

    data = np.load(path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    return x_train, y_train, x_test, y_test


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap NumPy arrays into PyTorch DataLoaders.
    """
    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute accuracy, precision, recall, F1 for binary classification.
    Uses F1 macro (as requested in the assignment), but since this is
    a binary task, macro vs binary will be very similar.
    """
    # convert to numpy
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
