"""Hand-checkable tests for src/analysis.py. Run with: python -m pytest tests/test_analysis.py -v"""
import numpy as np
import pytest

from src.analysis import accuracy_per_t, best_t, paired_bootstrap_ci


def test_accuracy_per_t() -> None:
    correct = np.array([[1, 0, 0, 0],      # t = 0: 1/4
                        [1, 1, 1, 0]],     # t = 1: 3/4
                       dtype=bool)
    assert accuracy_per_t(correct).tolist() == [0.25, 0.75]


def test_gain_is_difference_of_accuracies() -> None:
    a = np.array([1, 0, 0, 0], dtype=bool)  # 1/4
    b = np.array([1, 1, 1, 0], dtype=bool)  # 3/4
    gain, lo, hi = paired_bootstrap_ci(a, b, n_boot=2_000)
    assert gain == 0.5
    assert 0.0 <= lo <= gain <= hi <= 1.0   # per-image diffs are 0 or 1


def test_constant_difference_collapses_ci() -> None:
    a = np.zeros(10)
    b = np.full(10, 0.3)                     # every image gains exactly 0.3
    gain, lo, hi = paired_bootstrap_ci(a, b, n_boot=500)
    assert gain == pytest.approx(0.3) and lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_pairing_removes_shared_image_variance() -> None:
    # Image difficulty varies a lot but b = a + 0.1 for every image:
    # a paired CI is a point; an unpaired one would be wide.
    a = np.linspace(0, 1, 50)
    gain, lo, hi = paired_bootstrap_ci(a, a + 0.1, n_boot=500)
    assert hi - lo == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_reproducible_and_chunk_independent() -> None:
    # Continuous scores: with 0/1 scores every resample mean is a multiple of 1/N,
    # so two seeds can give identical percentile CIs.
    rng = np.random.default_rng(1)
    a, b = rng.random(30), rng.random(30)
    first = paired_bootstrap_ci(a, b, n_boot=1_000, seed=3)
    assert first == paired_bootstrap_ci(a, b, n_boot=1_000, seed=3)
    assert first != paired_bootstrap_ci(a, b, n_boot=1_000, seed=4)
    # 1,000 rows in chunks of 1,000 or 250 draw the same random numbers in the same order
    assert first == paired_bootstrap_ci(a, b, n_boot=1_000, seed=3, chunk=250)


def test_best_t_skips_t0_and_takes_first_tie() -> None:
    acc = np.array([0.9, 0.5, 0.7, 0.7, 0.6])
    assert best_t(acc) == 2          # t = 0 (0.9) excluded; tie at 2 and 3 -> 2
    assert best_t(acc, t_min=0) == 0
