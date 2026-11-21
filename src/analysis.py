"""Statistics over saved per-image results: accuracy per timestep, paired bootstrap CIs, best t.

All functions take numpy arrays so they can be rerun from the saved .npz files
without the model.
"""
from __future__ import annotations

import numpy as np


def accuracy_per_t(correct: np.ndarray) -> np.ndarray:
    """correct: [T+1, N] bool (image n correct at step t) -> accuracy [T+1] in [0, 1]."""
    assert correct.ndim == 2 and correct.shape[1] > 0
    return correct.astype(np.float64).mean(axis=1)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                        level: float = 0.95, chunk: int = 1_000) -> tuple[float, float, float]:
    """Gain mean(b) - mean(a) and its percentile bootstrap CI over images.

    a, b: [N] per-image scores of the SAME images under two conditions (e.g. top-1
    correct at t = 0 and t = 15). Each resample draws N image indices with
    replacement and uses them for both a and b (paired). Returns (gain, lo, hi).
    """
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    diff = b.astype(np.float64) - a.astype(np.float64)             # [N]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):                          # chunks keep idx at ~chunk*N ints
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))  # [size, N]
        means[start:start + size] = diff[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(diff.mean()), float(lo), float(hi)


def best_t(acc: np.ndarray, t_min: int = 1) -> int:
    """argmax over t >= t_min of acc [T+1]; ties go to the earliest t.

    Chosen on the same images it is reported on, so its gain is optimistic.
    """
    assert len(acc) > t_min
    return t_min + int(np.argmax(acc[t_min:]))
