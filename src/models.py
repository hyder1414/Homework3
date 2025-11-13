from typing import Literal

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Map a string to a PyTorch activation module.
    Allowed: 'relu', 'tanh', 'sigmoid'.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class SentimentRNN(nn.Module):
    """
    Generic sequence classifier that can behave as:
      - vanilla RNN
      - LSTM
      - Bidirectional LSTM

    It follows the HW3 design notes:
      - Embedding layer (dim=100)
      - Two fully-connected hidden layers (size=64) with chosen activation
      - Dropout between layers
      - Final sigmoid output for binary classification
    """

    def __init__(
        self,
        vocab_size: int,
        architecture: Literal["rnn", "lstm", "bilstm"] = "lstm",
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        embedding_dim: int = 100,
        rnn_hidden_dim: int = 64,
        fc_hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.architecture = architecture.lower()
        self.activation_name = activation.lower()

        # 1) Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,  # our PAD token
        )

        # 2) Recurrent block (RNN / LSTM / BiLSTM)
        if self.architecture == "rnn":
            rnn_cls = nn.RNN
            bidirectional = False
        elif self.architecture == "lstm":
            rnn_cls = nn.LSTM
            bidirectional = False
        elif self.architecture == "bilstm":
            rnn_cls = nn.LSTM
            bidirectional = True
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.bidirectional = bidirectional
        self.rnn_hidden_dim = rnn_hidden_dim

        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,          # keep RNN stack simple; we add FC hidden layers instead
            batch_first=True,
            bidirectional=bidirectional,
        )

        # 3) Fully connected layers (two hidden layers with configurable activation)
        rnn_out_dim = rnn_hidden_dim * (2 if bidirectional else 1)

        self.fc1 = nn.Linear(rnn_out_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc_out = nn.Linear(fc_hidden_dim, 1)

        self.activation = get_activation(self.activation_name)
        self.dropout = nn.Dropout(dropout)

        # Sigmoid at the very end for probabilities
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor of shape (batch_size, seq_len) with token IDs.
        Returns:
            probs: FloatTensor of shape (batch_size,) with values in [0, 1].
        """
        # x -> (B, L) -> embeddings (B, L, E)
        emb = self.embedding(x)

        if self.architecture == "lstm" or self.architecture == "bilstm":
            rnn_out, (h_n, c_n) = self.rnn(emb)
        else:
            rnn_out, h_n = self.rnn(emb)

        # Use output at the last time step (shape: B x (H * num_directions))
        last_output = rnn_out[:, -1, :]

        # Two-layer MLP with chosen activation + dropout
        h = self.fc1(last_output)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.activation(h)
        h = self.dropout(h)

        logits = self.fc_out(h).squeeze(1)   # (B,)
        probs = self.output_activation(logits)  # in [0,1]

        return probs
