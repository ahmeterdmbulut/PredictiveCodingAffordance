"""Tests for src/degrade.py. Run with: python -m pytest tests/test_degrade.py -v"""
import pytest
import torch

from src.degrade import SEED_STRIDE, add_gaussian_noise, noise_seed, seeded_gaussian_noise


def test_noise_seed_values() -> None:
    assert noise_seed(59) == 59
    assert noise_seed(59, seed=2) == 2 * SEED_STRIDE + 59
    with pytest.raises(AssertionError):
        noise_seed(SEED_STRIDE)


def test_same_id_same_noise_regardless_of_batch() -> None:
    a = seeded_gaussian_noise([7, 3, 11], shape=(3, 8, 8))   # [3, 3, 8, 8]
    b = seeded_gaussian_noise([11, 7], shape=(3, 8, 8))      # other batch, other order
    assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])


def test_matches_a_plain_generator() -> None:
    expected = torch.randn((3, 8, 8), generator=torch.Generator().manual_seed(42))
    assert torch.equal(seeded_gaussian_noise([42], shape=(3, 8, 8))[0], expected)


def test_different_ids_or_seeds_differ() -> None:
    eps = seeded_gaussian_noise([1, 2], shape=(3, 8, 8))
    assert not torch.equal(eps[0], eps[1])
    assert not torch.equal(eps[0], seeded_gaussian_noise([1], shape=(3, 8, 8), seed=1)[0])


def test_standard_normal_moments() -> None:
    eps = seeded_gaussian_noise([5], shape=(3, 224, 224))    # 150,528 samples
    assert eps.dtype == torch.float32
    assert abs(eps.mean().item()) < 0.01 and abs(eps.std().item() - 1) < 0.01


def test_add_noise_scales_and_does_not_clamp() -> None:
    x = torch.full((1, 3, 2, 2), 2.0)
    eps = torch.ones(1, 3, 2, 2)
    assert add_gaussian_noise(x, eps, 0.0) is x
    assert torch.equal(add_gaussian_noise(x, eps, 0.5), torch.full_like(x, 2.5))
    assert torch.equal(add_gaussian_noise(x, -eps, 10.0), torch.full_like(x, -8.0))  # no clamp
