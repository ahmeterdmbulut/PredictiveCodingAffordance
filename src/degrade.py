"""Test-time image degradations. Gate 1 needs only additive Gaussian noise.

Convention (CLAUDE.md; predify2021 manifold_projection/calculate_corr_dists.py
L75-79, L128): N(0, sigma) is added AFTER ImageNet Normalize, in normalized-image
units, without clamping. Upstream reseeds the global RNG every batch; here each
image gets its own CPU generator seeded from its id, so its noise does not depend
on batch size, batch order, device or PC configuration.
"""
from __future__ import annotations

from typing import Sequence

import torch

SEED_STRIDE = 100_000  # > every image id (ImageNet val_id <= 50,000), so (seed, id) never collide


def noise_seed(image_id: int, seed: int = 0) -> int:
    """Generator seed for one image: seed * SEED_STRIDE + image_id."""
    assert 0 <= image_id < SEED_STRIDE, f"image id {image_id} outside [0, {SEED_STRIDE})"
    return seed * SEED_STRIDE + image_id


def seeded_gaussian_noise(image_ids: Sequence[int], shape: tuple[int, ...] = (3, 224, 224),
                          seed: int = 0) -> torch.Tensor:
    """Standard normal noise eps [B, *shape], float32 on CPU, one generator per image."""
    eps = torch.empty(len(image_ids), *shape)
    for i, image_id in enumerate(image_ids):
        gen = torch.Generator().manual_seed(noise_seed(int(image_id), seed))
        eps[i] = torch.randn(shape, generator=gen)
    return eps


def add_gaussian_noise(x: torch.Tensor, eps: torch.Tensor, sigma: float) -> torch.Tensor:
    """x + sigma * eps, no clamping. x, eps: [B, 3, H, W] on the same device.

    The same eps is reused for every sigma, so noise levels differ only in scale.
    """
    assert x.shape == eps.shape, f"{x.shape} != {eps.shape}"
    if sigma == 0:
        return x
    return x + sigma * eps
