# Homework 3 – RNN-based Sentiment Classification on IMDb

This repository contains my solution for **MSML 641 – Homework 3**.

The goal is to build and compare several RNN-based models (RNN, LSTM, BiLSTM) for binary sentiment classification on the IMDb movie review dataset, and to study the effect of:

* Model architecture (RNN vs LSTM vs BiLSTM)
* Sequence length (25, 50, 100 tokens)
* Activation function (ReLU vs Tanh in the classifier head)
* Optimizer (Adam vs SGD)
* Stability strategy (with vs without gradient clipping)

All models are implemented in PyTorch and train on CPU.

---

## Repository Structure

```text
Homework3/
├── data/
│   ├── imdb_seq25.npz           # preprocessed data, seq_len=25
│   ├── imdb_seq50.npz           # preprocessed data, seq_len=50
│   ├── imdb_seq100.npz          # preprocessed data, seq_len=100
│   ├── vocab.json               # token → index mapping
│   └── dataset_stats.json       # length statistics and metadata
├── results/
│   ├── README.md                # short description of result files
│   ├── metrics.csv              # per-epoch metrics for all runs
│   ├── summary_best_runs.csv    # one row per configuration (best test F1)
│   ├── best_model_*.pt          # best model weights per run
│   └── plots/
│       ├── accuracy_f1_vs_seq_len.png
│       ├── training_loss_best.png
│       └── training_loss_worst.png
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # preprocessing + NPZ generation
│   ├── models.py                # SentimentRNN (RNN / LSTM / BiLSTM)
│   ├── utils.py                 # dataset, dataloaders, metrics, helpers
│   ├── train.py                 # training loop and experiment config
│   └── evaluate.py              # result aggregation and plotting
├── .gitignore
├── requirements.txt
└── report.pdf                   # written report for the assignment
```

Note: the raw IMDb CSV file (`imdb_reviews.csv`) is not versioned in this repository to avoid large-file issues. See the Data section below for instructions.

---

## Requirements and Setup

Tested with:

* Python 3.11
* CPU-only machine (no GPU required)

Install dependencies in a virtual environment (recommended):

```bash
cd /path/to/Homework3

python3 -m venv .venv
source .venv/bin/activate        # on macOS / Linux

pip install -r requirements.txt
```

---

## Data

The assignment uses the **IMDb Movie Review** dataset (50,000 labeled reviews: 25k positive, 25k negative).

1. Download the dataset CSV (e.g., `IMDB Dataset.csv`) from the course or Kaggle.
2. Place it into the `data/` directory.
3. Rename it to:

```text
data/imdb_reviews.csv
```

This file is listed in `.gitignore`, so it will not be pushed to GitHub.

After this, the preprocessing script can find and process the data.

---

## Step 1: Preprocessing

Preprocessing is implemented in `src/preprocess.py`. It:

* Loads `data/imdb_reviews.csv`
* Lowercases and cleans the review text
* Tokenizes the text
* Builds a vocabulary of the 10,000 most frequent words (plus PAD token)
* Creates a stratified 50/50 train–test split (25,000 / 25,000)
* Converts reviews to sequences of token IDs
* Pads/truncates sequences to the following lengths: 25, 50, 100
* Saves NumPy arrays for each sequence length into `.npz` files
* Saves dataset statistics and vocabulary

Run:

```bash
cd /path/to/Homework3
python src/preprocess.py
```

This will create:

```text
data/
├── imdb_seq25.npz
├── imdb_seq50.npz
├── imdb_seq100.npz
├── vocab.json
└── dataset_stats.json
```

You need to run this once after placing `imdb_reviews.csv` in `data/`.

---

## Step 2: Model and Training Configuration

Training is controlled via configuration variables at the top of `src/train.py`:

```python
ARCHITECTURE   = "lstm"     # "rnn", "lstm", or "bilstm"
ACTIVATION     = "relu"     # "relu" or "tanh" (classifier head)
OPTIMIZER_NAME = "adam"     # "adam" or "sgd"
SEQ_LEN        = 100        # 25, 50, or 100
USE_GRAD_CLIP  = True       # True / False
MAX_GRAD_NORM  = 5.0 if USE_GRAD_CLIP else None

BATCH_SIZE     = 32
NUM_EPOCHS     = 5
```

The `SentimentRNN` model in `src/models.py` includes:

* Embedding layer (embedding dim 100)
* Recurrent layer:

  * vanilla RNN, LSTM, or BiLSTM (hidden size 64)
* Two-layer fully connected classifier with dropout
* Sigmoid output for binary classification

All hyperparameters other than the factor under study (architecture, sequence length, activation, optimizer, gradient clipping) are kept fixed to enable controlled comparisons.

---

## Step 3: Training Experiments

To launch a training run, after creating the venv and running preprocessing:

```bash
cd /path/to/Homework3
python -m src.train
```

This will:

* Load the correct `data/imdb_seq{SEQ_LEN}.npz` based on `SEQ_LEN`
* Train for `NUM_EPOCHS` on the training split
* Evaluate on the test set after each epoch
* Log metrics (loss, accuracy, precision, recall, F1, epoch time) into `results/metrics.csv`
* Save the best model checkpoint (max test F1) as:

```text
results/best_model_<run_name>.pt
```

The run name is automatically constructed from the configuration, for example:

* `rnn_relu_adam_L50_clip`
* `lstm_relu_adam_L25_clip`
* `lstm_relu_adam_L50_clip`
* `lstm_relu_adam_L100_clip`
* `bilstm_relu_adam_L50_clip`
* `lstm_relu_sgd_L50_clip`
* `lstm_tanh_adam_L50_noclip`
* `lstm_relu_adam_L50_noclip`

On a CPU-only laptop, each run typically takes a few minutes, depending on the architecture and sequence length.

---

## Step 4: Evaluation and Plots

After running one or more experiments, summarize results and generate plots:

```bash
cd /path/to/Homework3
python -m src.evaluate
```

This script:

1. Reads `results/metrics.csv`
2. Computes the best test metrics (accuracy, F1, epoch time) per configuration
3. Writes a summary CSV:

```text
results/summary_best_runs.csv
```

4. Produces plots in `results/plots/`:

* `accuracy_f1_vs_seq_len.png`

  * Accuracy and F1 vs sequence length for LSTM + ReLU + Adam + clipping.
* `training_loss_best.png`

  * Training loss vs epochs for the best overall model.
* `training_loss_worst.png`

  * Training loss vs epochs for the worst-performing model (based on F1).

These are the plots used and discussed in the report.

---

## Key Configurations Used in the Report

The main configurations used for the comparative study are:

* **Architecture comparison** (L=50, ReLU, Adam, clip):

  * `rnn_relu_adam_L50_clip`
  * `lstm_relu_adam_L50_clip`
  * `bilstm_relu_adam_L50_clip`

* **Sequence length comparison** (LSTM, ReLU, Adam, clip):

  * `lstm_relu_adam_L25_clip`
  * `lstm_relu_adam_L50_clip`
  * `lstm_relu_adam_L100_clip`

* **Activation comparison** (LSTM, L=50, Adam, no clip):

  * `lstm_relu_adam_L50_noclip`
  * `lstm_tanh_adam_L50_noclip`

* **Optimizer and clipping comparison**:

  * `lstm_relu_adam_L50_clip` vs `lstm_relu_adam_L50_noclip` (effect of gradient clipping)
  * `lstm_relu_sgd_L50_clip` vs `lstm_relu_adam_L50_clip` (Adam vs SGD)

The best-performing configuration overall is:

* `lstm_relu_adam_L100_clip`

  * Architecture: LSTM
  * Activation: ReLU
  * Optimizer: Adam
  * Sequence length: 100
  * Gradient clipping: enabled (max norm 5.0)

---

## How to Reproduce the Best Model

1. Place `imdb_reviews.csv` in `data/` and run:

   ```bash
   python src/preprocess.py
   ```

2. In `src/train.py`, set:

   ```python
   ARCHITECTURE   = "lstm"
   ACTIVATION     = "relu"
   OPTIMIZER_NAME = "adam"
   SEQ_LEN        = 100
   USE_GRAD_CLIP  = True
   MAX_GRAD_NORM  = 5.0
   ```

3. Run training:

   ```bash
   python -m src.train
   ```

4. (Optional) Summarize and plot:

   ```bash
   python -m src.evaluate
   ```

This recreates the `lstm_relu_adam_L100_clip` run, which achieved the highest test performance in the experiments.

---

## Notes

* Random seeds (NumPy, PyTorch, and Python’s `random`) are set in the code for reproducibility.
* All paths in the scripts are relative to the project root (the directory containing `src/`, `data/`, and `results/`).
* The written report (`report.pdf`) provides a detailed description of the methods, experiments, and findings.
