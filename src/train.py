import os
import csv
import time
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop

from .models import SentimentRNN
from .utils import load_imdb_data, create_dataloaders, compute_metrics


# ------------------------------------------------
# Config: edit these for each experiment you run
# ------------------------------------------------
ARCHITECTURE = "lstm"          # "rnn", "lstm", "bilstm"
ACTIVATION = "tanh"            # "relu", "tanh", "sigmoid"
OPTIMIZER_NAME = "adam"        # "adam", "sgd", "rmsprop"
SEQ_LEN = 50                   # 25, 50, or 100
USE_GRAD_CLIP = False           # True or False
MAX_GRAD_NORM = 5.0 if USE_GRAD_CLIP else None

BATCH_SIZE = 32                # as per HW spec
LR = 1e-3
NUM_EPOCHS = 5                 # you can adjust based on time
DROPOUT = 0.5                  # between 0.3 and 0.5 as suggested

RUN_NAME = f"{ARCHITECTURE}_{ACTIVATION}_{OPTIMIZER_NAME}_L{SEQ_LEN}_" \
           f"{'clip' if USE_GRAD_CLIP else 'noclip'}"


# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon / Metal
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr)
    elif name == "sgd":
        return SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device,
    max_grad_norm: Optional[float] = None,
):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.float().to(device)  # BCELoss expects float targets

        optimizer.zero_grad()
        probs = model(xb)          # (batch,)
        loss = criterion(probs, yb)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = (probs >= 0.5).long().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.long().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(train_loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)

    return avg_loss, metrics


def evaluate(
    model: nn.Module,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)

            probs = model(xb)
            loss = criterion(probs, yb)

            total_loss += loss.item() * xb.size(0)

            preds = (probs >= 0.5).long().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.long().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(data_loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)

    return avg_loss, metrics


def ensure_metrics_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_name",
                    "architecture",
                    "activation",
                    "optimizer",
                    "seq_len",
                    "grad_clipping",
                    "epoch",
                    "split",
                    "loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "epoch_time_sec",
                ]
            )


def append_metrics_row(
    path: str,
    run_name: str,
    architecture: str,
    activation: str,
    optimizer_name: str,
    seq_len: int,
    grad_clipping: bool,
    epoch: int,
    split: str,
    loss: float,
    metrics: dict,
    epoch_time_sec: float,
):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                run_name,
                architecture,
                activation,
                optimizer_name,
                seq_len,
                grad_clipping,
                epoch,
                split,
                f"{loss:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{epoch_time_sec:.2f}",
            ]
        )


# ------------------------------------------------
# Main training entry point
# ------------------------------------------------
def main():
    set_seeds(42)
    device = get_device()
    print(f"Using device: {device}")

    print(f"\nConfiguration:")
    print(f"  ARCHITECTURE : {ARCHITECTURE}")
    print(f"  ACTIVATION   : {ACTIVATION}")
    print(f"  OPTIMIZER    : {OPTIMIZER_NAME}")
    print(f"  SEQ_LEN      : {SEQ_LEN}")
    print(f"  GRAD CLIP    : {USE_GRAD_CLIP} (max_norm={MAX_GRAD_NORM})")
    print(f"  BATCH_SIZE   : {BATCH_SIZE}")
    print(f"  NUM_EPOCHS   : {NUM_EPOCHS}")
    print(f"  RUN_NAME     : {RUN_NAME}")

    # ---- Load data ----
    x_train, y_train, x_test, y_test = load_imdb_data(SEQ_LEN)
    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE
    )

    # Infer vocab size from data (max token id + 1)
    vocab_size = int(max(x_train.max(), x_test.max())) + 1
    print(f"\nInferred vocab_size (including PAD): {vocab_size}")

    # ---- Build model, loss, optimizer ----
    model = SentimentRNN(
        vocab_size=vocab_size,
        architecture=ARCHITECTURE,
        activation=ACTIVATION,
        embedding_dim=100,
        rnn_hidden_dim=64,
        fc_hidden_dim=64,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = get_optimizer(OPTIMIZER_NAME, model.parameters(), LR)

    metrics_csv_path = os.path.join("results", "metrics.csv")
    ensure_metrics_csv(metrics_csv_path)

    # ---- Training loop ----
    best_test_f1 = 0.0
    best_model_path = os.path.join("results", f"best_model_{RUN_NAME}.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        start_time = time.perf_counter()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, MAX_GRAD_NORM
        )
        epoch_time = time.perf_counter() - start_time

        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

        print(
            f"  Train: loss={train_loss:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}"
        )
        print(
            f"  Test : loss={test_loss:.4f}, "
            f"acc={test_metrics['accuracy']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"epoch_time={epoch_time:.2f}s"
        )

        # log to CSV (train + test rows)
        append_metrics_row(
            metrics_csv_path,
            RUN_NAME,
            ARCHITECTURE,
            ACTIVATION,
            OPTIMIZER_NAME,
            SEQ_LEN,
            USE_GRAD_CLIP,
            epoch,
            "train",
            train_loss,
            train_metrics,
            epoch_time,
        )
        append_metrics_row(
            metrics_csv_path,
            RUN_NAME,
            ARCHITECTURE,
            ACTIVATION,
            OPTIMIZER_NAME,
            SEQ_LEN,
            USE_GRAD_CLIP,
            epoch,
            "test",
            test_loss,
            test_metrics,
            epoch_time,
        )

        # Track best model by test F1
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "ARCHITECTURE": ARCHITECTURE,
                        "ACTIVATION": ACTIVATION,
                        "OPTIMIZER_NAME": OPTIMIZER_NAME,
                        "SEQ_LEN": SEQ_LEN,
                        "USE_GRAD_CLIP": USE_GRAD_CLIP,
                        "MAX_GRAD_NORM": MAX_GRAD_NORM,
                        "BATCH_SIZE": BATCH_SIZE,
                        "LR": LR,
                        "DROPOUT": DROPOUT,
                    },
                },
                best_model_path,
            )
            print(f"  ✅ New best model saved to {best_model_path} (F1={best_test_f1:.4f})")

    print(f"\nTraining finished. Best test F1 for this run: {best_test_f1:.4f}")
    print(f"Metrics logged to {metrics_csv_path}")
    print(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
