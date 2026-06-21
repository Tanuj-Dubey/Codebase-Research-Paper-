"""
slm_rul_model.py
================
Small Language Model (SLM) for Battery RUL Prediction.

Architecture:
    Input: sliding window of W=20 cycles x 5 HIS features
    -> Linear embedding (5 -> d_model=64)
    -> Sinusoidal positional encoding
    -> 2x Transformer Encoder layers (4 heads, d_ff=128)
    -> Global average pooling
    -> FC: 64 -> 32 -> 1  (sigmoid output -> RUL in [0,1])
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── SLM Transformer RUL Model ─────────────────────────────────────────────────

class SLMTransformerRUL(nn.Module):
    """
    Compact Transformer Encoder for battery RUL regression.

    Parameters
    ----------
    n_features  : int  - number of input features per time step (default 5)
    d_model     : int  - embedding dimension (default 64)
    n_heads     : int  - number of attention heads (default 4)
    d_ff        : int  - feed-forward hidden size (default 128)
    n_layers    : int  - number of encoder layers (default 2)
    window_size : int  - input sequence length (default 20)
    dropout     : float- dropout rate (default 0.1)
    """

    def __init__(
        self,
        n_features:  int   = 5,
        d_model:     int   = 64,
        n_heads:     int   = 4,
        d_ff:        int   = 128,
        n_layers:    int   = 2,
        window_size: int   = 20,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=window_size + 10,
                                             dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = n_heads,
            dim_feedforward = d_ff,
            dropout    = dropout,
            batch_first = True,
            norm_first  = True,     # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),           # RUL in [0, 1]
        )

    def summary(self):
        """Returns a string summary of the model architecture."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        s = "SLMTransformerRUL Summary:\n"
        s += "--------------------------\n"
        s += f"  Total Trainable Params : {n_params:,}\n"
        s += f"  Input Proj             : Linear({self.input_proj.in_features}, {self.input_proj.out_features})\n"
        s += f"  Transformer Encoder    : {len(self.encoder.layers)} layers, {self.encoder.layers[0].self_attn.num_heads} heads\n"
        s += f"  FC Head                : {list(self.head.children())}\n"
        s += "--------------------------"
        return s

    def forward(self, x):
        # x: (batch, window_size, n_features)
        x = self.input_proj(x)      # -> (batch, W, d_model)
        x = self.pos_enc(x)         # add positional encoding
        x = self.encoder(x)         # -> (batch, W, d_model)
        x = x.mean(dim=1)           # global average pooling -> (batch, d_model)
        return self.head(x).squeeze(-1)   # -> (batch,)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RULDataset(Dataset):
    """
    Sliding window dataset for RUL prediction.

    Parameters
    ----------
    features    : np.ndarray  shape (N_cycles, n_features) - normalised
    rul_labels  : np.ndarray  shape (N_cycles,)             - [0, 1]
    window_size : int
    """

    def __init__(self, features: np.ndarray, rul_labels: np.ndarray,
                 window_size: int = 20):
        self.window_size = window_size
        self.X, self.y  = self._create_windows(features, rul_labels)

    def _create_windows(self, features, labels):
        X, y = [], []
        n = len(features)
        for i in range(n - self.window_size + 1):
            X.append(features[i : i + self.window_size])
            y.append(labels[i + self.window_size - 1])
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Quick unit test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SLMTransformerRUL()
    x = torch.randn(8, 20, 5)
    y = model(x)
    assert y.shape == (8,), f"Expected (8,), got {y.shape}"
    assert (y >= 0).all() and (y <= 1).all(), "Output not in [0,1]"
    print(model.summary())
    print("SLMTransformerRUL: OK")
    print("  Output shape : {}".format(y.shape))
    print("  Output range : [{:.4f}, {:.4f}]".format(y.min().item(), y.max().item()))
