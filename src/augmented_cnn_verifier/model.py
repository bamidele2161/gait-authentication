"""Small four-block 1D CNN following the architecture reported by Choi et al."""

import torch
from torch import nn

from src.augmented_cnn_verifier.config import CHANNELS, WINDOW_SAMPLES


class PaperCNNVerifier(nn.Module):
    """Return one legitimate-user logit for a two-channel magnitude window."""

    def __init__(self, filters=6, kernel_size=6, dropout=0.30):
        super().__init__()
        if filters < 1 or kernel_size < 1:
            raise ValueError("filters and kernel_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.features = nn.Sequential(
            nn.Conv1d(CHANNELS, filters, kernel_size),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(filters, filters, kernel_size),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(filters, filters, kernel_size),
            nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(filters, filters, kernel_size),
            nn.ReLU(), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, CHANNELS, WINDOW_SAMPLES)).numel()
        self.output = nn.Linear(flattened, 1)

    def forward(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (WINDOW_SAMPLES, CHANNELS):
            raise ValueError(
                f"Expected (batch, {WINDOW_SAMPLES}, {CHANNELS}), got {tuple(windows.shape)}"
            )
        features = self.features(windows.transpose(1, 2)).flatten(1)
        return self.output(features).squeeze(1)

