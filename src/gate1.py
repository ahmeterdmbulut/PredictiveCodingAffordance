"""Gate 1 positive control: run PVGG16 t = 0..T on noisy ImageNet val images, record per image.

One call of run_sigma = one backbone pass per (image, sigma) for one PC configuration.
Per-image records are saved so every statistic can be recomputed from disk (src/analysis.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analysis import accuracy_per_t, best_t, paired_bootstrap_ci
from src.degrade import add_gaussian_noise, seeded_gaussian_noise
from src.load_predify import N_PCODERS, iterate_timesteps

# Choksi 2021 Table S4, PVGG16 top-1 (%), (t = 0, t = 15), "fixed" HP set.
PAPER_TABLE_S4 = {0.0: (71.63, 71.47), 0.5: (35.61, 38.59), 0.75: (16.69, 18.46), 1.0: (5.59, 7.05)}


@dataclass
class SigmaRecords:
    """Per-image results for one sigma; arrays are [T+1, N] unless noted."""
    val_id: np.ndarray        # [N] int64
    label: np.ndarray         # [N] int64
    pred_top1: np.ndarray     # int16 predicted class
    top1_correct: np.ndarray  # bool
    top5_correct: np.ndarray  # bool
    true_logit: np.ndarray    # float32 logit of the true class


@dataclass
class SigmaRun:
    records: SigmaRecords
    seconds: float
    vgg16_max_abs_diff: float | None       # t = 0 logits vs plain VGG16, sigma = 0 only
    drift: np.ndarray | None               # [T+1, n_drift, 5] ||rep(t) - rep(0)|| / ||rep(0)||
    drift_val_ids: np.ndarray | None       # [n_drift]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _per_image_norm(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B] Frobenius norm per image."""
    return torch.linalg.vector_norm(x.flatten(1), dim=1)


@torch.no_grad()
def run_sigma(net: nn.Module, loader: DataLoader, sigma: float, T: int, device: torch.device,
              noise_seed: int = 0, vgg16: nn.Module | None = None,
              drift_ids: set[int] | None = None) -> SigmaRun:
    """Run every image of `loader` (yielding x [B,3,224,224], label [B], val_id [B]) at one sigma.

    vgg16: plain torchvision VGG16; if given, t = 0 logits are compared with it on the same input.
    drift_ids: val_ids whose per-PCoder drift is logged (rep(0) kept only for those rows).
    """
    n = len(loader.dataset)
    rec = SigmaRecords(
        val_id=np.zeros(n, np.int64), label=np.zeros(n, np.int64),
        pred_top1=np.zeros((T + 1, n), np.int16), top1_correct=np.zeros((T + 1, n), bool),
        top5_correct=np.zeros((T + 1, n), bool), true_logit=np.zeros((T + 1, n), np.float32))
    drift_rows: list[torch.Tensor] = []     # each [T+1, b_drift, 5]
    drift_vids: list[int] = []
    max_diff = 0.0
    start, row = time.perf_counter(), 0
    for x, y, vid in loader:                # x: [B, 3, 224, 224], y, vid: [B]
        b = len(y)
        eps = seeded_gaussian_noise(vid.tolist(), shape=tuple(x.shape[1:]), seed=noise_seed)
        x = add_gaussian_noise(x, eps, sigma).to(device)
        y_dev = y.to(device)
        sel = torch.tensor([int(v) in drift_ids for v in vid]) if drift_ids else torch.zeros(b, dtype=torch.bool)
        rows = slice(row, row + b)
        rec.val_id[rows], rec.label[rows] = vid.numpy(), y.numpy()
        ref, ref_norm, batch_drift = None, None, []
        for t, logits, reps in iterate_timesteps(net, x, T):   # logits [B, 1000]
            top5 = logits.topk(5, dim=1).indices                # [B, 5]
            rec.pred_top1[t, rows] = top5[:, 0].cpu().numpy()
            rec.top1_correct[t, rows] = (top5[:, 0] == y_dev).cpu().numpy()
            rec.top5_correct[t, rows] = (top5 == y_dev[:, None]).any(dim=1).cpu().numpy()
            rec.true_logit[t, rows] = logits.gather(1, y_dev[:, None])[:, 0].cpu().numpy()
            if t == 0 and vgg16 is not None:
                max_diff = max(max_diff, (logits - vgg16(x)).abs().max().item())
            if sel.any():
                cur = [r[sel.to(r.device)] for r in reps]       # 5 x [b_drift, C, H, W]
                if t == 0:
                    ref, ref_norm = cur, [_per_image_norm(r) for r in cur]
                batch_drift.append(torch.stack(
                    [_per_image_norm(c - r0) / n0 for c, r0, n0 in zip(cur, ref, ref_norm)], dim=1))
        if sel.any():
            drift_rows.append(torch.stack(batch_drift).cpu())    # [T+1, b_drift, 5]
            drift_vids += [int(v) for v in vid[sel]]
        row += b
    _sync(device)
    assert row == n
    drift = torch.cat(drift_rows, dim=1).numpy() if drift_rows else None
    assert drift is None or drift.shape[2] == N_PCODERS
    return SigmaRun(rec, time.perf_counter() - start, max_diff if vgg16 is not None else None,
                    drift, np.array(drift_vids, np.int64) if drift_rows else None)


def sigma_metrics(rec: SigmaRecords, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Accuracies per t (fractions), gain t = T vs 0 with paired bootstrap CI, best t and its gain."""
    top1 = accuracy_per_t(rec.top1_correct)                    # [T+1]
    top5 = accuracy_per_t(rec.top5_correct)
    T = len(top1) - 1
    gain, lo, hi = paired_bootstrap_ci(rec.top1_correct[0], rec.top1_correct[T], n_boot, seed)
    tb = best_t(top1)
    b_gain, b_lo, b_hi = paired_bootstrap_ci(rec.top1_correct[0], rec.top1_correct[tb], n_boot, seed)
    return {
        "n_images": int(rec.top1_correct.shape[1]),
        "top1": top1.tolist(), "top5": top5.tolist(),
        "gain_T": {"t": T, "gain": gain, "ci95": [lo, hi]},
        "best_t": {"t": tb, "gain": b_gain, "ci95": [b_lo, b_hi],
                   "note": "t chosen on the same images (in-sample, optimistic)"},
    }


def check_pass_rule(metrics: dict[float, dict], paper: dict[float, tuple[float, float]] = PAPER_TABLE_S4) -> dict:
    """Gate 1 pass rule, fixed 2026-11-20 before any full run. All values in percentage points.

    metrics[sigma] = sigma_metrics(...) output (fractions). Rules:
    (a) |acc(0) - paper(0)| <= 3 at every sigma;
    (b) sigma 0.5: gain CI lower bound > 0 and gain in [0.5, 2] x paper gain;
    (c) sigma 0.75 and 1.0: gain > 0;
    (d) sigma 0: |gain| <= 1.
    """
    pp = {s: {"acc0": 100 * m["top1"][0], "gain": 100 * m["gain_T"]["gain"], "lo": 100 * m["gain_T"]["ci95"][0]}
          for s, m in metrics.items()}
    paper_gain_05 = paper[0.5][1] - paper[0.5][0]
    t0_diff = {s: pp[s]["acc0"] - paper[s][0] for s in paper}
    rules = {
        "a": {"pass": all(abs(d) <= 3 for d in t0_diff.values()),
              "detail": {f"{s:.2f}": round(d, 2) for s, d in t0_diff.items()},
              "text": "|acc(t=0) - paper(t=0)| <= 3 pp at every sigma"},
        "b": {"pass": pp[0.5]["lo"] > 0 and 0.5 * paper_gain_05 <= pp[0.5]["gain"] <= 2 * paper_gain_05,
              "detail": {"gain": round(pp[0.5]["gain"], 2), "ci_low": round(pp[0.5]["lo"], 2),
                         "allowed": [round(0.5 * paper_gain_05, 2), round(2 * paper_gain_05, 2)]},
              "text": "sigma 0.5: CI lower bound > 0 and gain in [0.5x, 2x] paper gain"},
        "c": {"pass": pp[0.75]["gain"] > 0 and pp[1.0]["gain"] > 0,
              "detail": {"0.75": round(pp[0.75]["gain"], 2), "1.00": round(pp[1.0]["gain"], 2)},
              "text": "sigma 0.75 and 1.0: gain > 0"},
        "d": {"pass": abs(pp[0.0]["gain"]) <= 1, "detail": {"gain": round(pp[0.0]["gain"], 2)},
              "text": "sigma 0: |gain| <= 1 pp"},
    }
    return {"rules": rules, "pass": all(r["pass"] for r in rules.values())}


def drift_summary(drift: np.ndarray) -> dict:
    """drift [T+1, n, 5] -> per PCoder, per t: mean and median over images."""
    return {f"pcoder{k + 1}": {"mean": drift[:, :, k].mean(axis=1).tolist(),
                               "median": np.median(drift[:, :, k], axis=1).tolist()}
            for k in range(drift.shape[2])}
