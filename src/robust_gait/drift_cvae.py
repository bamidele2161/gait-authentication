"""Conditional VAE that learns the distribution of condition drift.

The drift bank holds one vector per development person and condition (mean
later stride minus mean ST-control stride), only 36 points per fold.  The CVAE
instead learns a generative model p(drift | condition) from many noisier
drift samples and then draws as many new drift vectors as needed, including
combinations that no single development person showed.

Training samples: for each person and later condition, the later recording is
cut into consecutive blocks of ``BLOCK`` strides, and each block's mean stride
minus the person's ST-control mean is one sample (~35 per person and
condition).  Sensors dead in either recording are zeroed, exactly as in the
drift bank.  Samples are compressed by PCA (fitted on the same people), the
CVAE works on standardized PCA scores, and generated scores are mapped back to
full (128, channels) stride offsets.

The CVAE is conditioned on the condition only (one-hot), not on the person:
the drift bank analysis showed that individual drift is not predictable from
12 people, so the model learns where and how much each condition moves a
stride.  Only ``people`` (never the enrolled user) are used.
"""

from __future__ import annotations

import hashlib
import zlib

import numpy as np

from src.robust_gait.data import SRC_DIR

CACHE_DIR = SRC_DIR / "data" / "processed" / "drift_cvae"
BLOCK = 10             # strides averaged per drift sample
COMPONENTS = 32         # PCA components modelled by the CVAE
LATENT = 8
HIDDEN = 128
EPOCHS = 400
BATCH = 128
LEARNING_RATE = 1e-3
BETA = 1.0              # KL weight
SAMPLES_PER_CONDITION = 30


def cache_path(people, seed):
    """Cache file for one training population; hyperparameters are in the key."""

    settings = (BLOCK, COMPONENTS, LATENT, HIDDEN, EPOCHS, BATCH, LEARNING_RATE, BETA,
                SAMPLES_PER_CONDITION)
    key = f"{','.join(sorted(people))}|{seed}|{settings}"
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npy"


def block_drifts(people, data, conditions, faulty_mask):
    """(samples, 128, channels) block drifts and their condition indices."""

    samples, labels = [], []
    for p in people:
        base = data[p, "st_control"].strides
        base_mean = base.mean(axis=0)
        for k, condition in enumerate(conditions):
            later = data[p, condition].strides
            mask = faulty_mask(base, later)
            for start in range(0, len(later) - BLOCK + 1, BLOCK):
                vector = later[start:start + BLOCK].mean(axis=0) - base_mean
                vector[:, mask] = 0.0
                samples.append(vector)
                labels.append(k)
    return np.stack(samples).astype(np.float32), np.asarray(labels)


def generate(people, data, conditions, faulty_mask, seed):
    """Train a CVAE on ``people`` and return generated drift vectors.

    Returns (len(conditions) * SAMPLES_PER_CONDITION, 128, channels).
    """

    import torch
    from torch import nn

    torch.set_num_threads(1)
    # Stable seed per training population, so every run reproduces the bank.
    key = zlib.crc32(",".join(sorted(people)).encode()) + seed
    torch.manual_seed(key)
    rng = np.random.default_rng(key)

    x, labels = block_drifts(people, data, conditions, faulty_mask)
    shape = x.shape[1:]
    flat = x.reshape(len(x), -1).astype(np.float64)
    mean = flat.mean(axis=0)
    _, singular, vt = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vt[:COMPONENTS]
    scores = (flat - mean) @ basis.T
    scale = scores.std(axis=0) + 1e-9
    z_scores = torch.tensor(scores / scale, dtype=torch.float32)
    onehot = torch.eye(len(conditions))[torch.tensor(labels)]

    class CVAE(nn.Module):
        def __init__(self):
            super().__init__()
            c = len(conditions)
            self.encoder = nn.Sequential(
                nn.Linear(COMPONENTS + c, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2 * LATENT),
            )
            self.decoder = nn.Sequential(
                nn.Linear(LATENT + c, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, COMPONENTS),
            )

        def forward(self, v, c):
            mu, logvar = self.encoder(torch.cat([v, c], 1)).chunk(2, 1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decoder(torch.cat([z, c], 1)), mu, logvar

    model = CVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n = len(z_scores)
    for _ in range(EPOCHS):
        order = torch.randperm(n)
        for i in range(0, n, BATCH):
            batch = order[i:i + BATCH]
            recon, mu, logvar = model(z_scores[batch], onehot[batch])
            reconstruction = ((recon - z_scores[batch]) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
            loss = reconstruction + BETA * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        c = torch.eye(len(conditions)).repeat_interleave(SAMPLES_PER_CONDITION, 0)
        z = torch.tensor(rng.standard_normal((len(c), LATENT)), dtype=torch.float32)
        generated = model.decoder(torch.cat([z, c], 1)).numpy().astype(np.float64)
    vectors = (generated * scale) @ basis + mean
    return vectors.reshape(len(vectors), *shape).astype(np.float32)
