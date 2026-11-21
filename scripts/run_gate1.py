"""Gate 1 positive control: one PC configuration on the Gate 1 ImageNet val subset.

Gaussian noise N(0, sigma) in normalized-image units (no clamping), one seeded
generator per image (src/degrade.py). PVGG16 is run for t = 0..T per sigma.

Usage (repo root):
  python scripts/run_gate1.py --hp-set fixed --pcoder-type PCoderN --device cuda --drift
  python scripts/run_gate1.py --hp-set fixed --pcoder-type PCoderN --n-images 64 --drift   # smoke run
Writes results/gate1_<hp>_<pcoder>_n<N>_<timestamp>/ with config.json, predictions_sigma<s>.npz
(one per sigma, written as soon as that sigma finishes), metrics.json, timing.json and,
with --drift, drift.npz + drift_summary.json (sigma = 0, every --drift-stride-th image).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import VGG16_Weights, vgg16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degrade import SEED_STRIDE  # noqa: E402
from src.gate1 import PAPER_TABLE_S4, drift_summary, run_sigma, sigma_metrics  # noqa: E402
from src.imagenet_val import GATE1_SUBSET_PATH, ImageNetSubset  # noqa: E402
from src.load_predify import (  # noqa: E402
    DEFAULT_C_SQRT_PATH, HP_SETS, build_pvgg16, c_sqrt_values, get_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hp-set", choices=["fixed", "tuned"], required=True)
    p.add_argument("--pcoder-type", choices=["PCoderN", "PCoder"], required=True)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    p.add_argument("--n-images", type=int, default=None, help="first N images of the subset (default: all)")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--drift", action="store_true", help="log PCoder drift at sigma = 0")
    p.add_argument("--drift-stride", type=int, default=10, help="every n-th image of the full subset")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights/predify/weights_pvgg16_imagenet")
    p.add_argument("--external-dir", type=Path, default=ROOT / "external")
    p.add_argument("--out-root", type=Path, default=ROOT / "results")
    return p.parse_args()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sigma_tag(sigma: float) -> str:
    return f"{sigma:.2f}"


def save_records(out_dir: Path, sigma: float, run) -> None:
    np.savez_compressed(out_dir / f"predictions_sigma{sigma_tag(sigma)}.npz", **asdict(run.records))


def gpu_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    return {"max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9}


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    dataset = ImageNetSubset(GATE1_SUBSET_PATH)
    n = len(dataset) if args.n_images is None else args.n_images
    subset = Subset(dataset, list(range(n)))
    assert n % args.batch_size == 0, "use full batches (plain PCoder's step depends on batch size)"
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    drift_ids = {dataset.images[i].val_id for i in range(0, n, args.drift_stride)} if args.drift else None

    net = build_pvgg16(weights_dir=args.weights_dir, external_dir=args.external_dir,
                       pcoder_type=args.pcoder_type, hp_set=args.hp_set, device=device)
    ref_vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / f"gate1_{args.hp_set}_{args.pcoder_type}_n{n}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device), "n_images": n,
        "subset": GATE1_SUBSET_PATH.relative_to(ROOT).as_posix(),
        "val_ids": [dataset.images[i].val_id for i in range(n)],
        "hp_values": HP_SETS[args.hp_set], "c_sqrt": c_sqrt_values(net),
        "c_sqrt_file": DEFAULT_C_SQRT_PATH.relative_to(ROOT).as_posix() if args.pcoder_type == "PCoderN" else None,
        "preprocessing": "Resize(224) + CenterCrop(224) + ImageNet Normalize (Predify)",
        "noise": (f"x + sigma * eps after Normalize, no clamping; eps ~ N(0,1) [3,224,224] from "
                  f"torch.Generator(CPU).manual_seed(noise_seed * {SEED_STRIDE} + val_id); "
                  "same eps for every sigma and configuration"),
        "batch_note": ("plain PCoder's error step scales with 1/batch size (MSE mean over the batch), "
                       "PCoderN's does not (prd.numel() / C_sqrt); all batches are full"),
        "paper_table_s4": {sigma_tag(s): v for s, v in PAPER_TABLE_S4.items()},
        "upstream_commits": {name: git_head(args.external_dir / name) for name in ("predify", "predify2021")},
        "torch": torch.__version__, "platform": platform.platform(), "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cudnn_benchmark": False,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Writing to {out_dir}", flush=True)

    metrics, timing = {}, {"per_sigma": {}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for sigma in args.sigmas:
        use_drift = drift_ids if sigma == 0 else None
        run = run_sigma(net, loader, sigma, args.T, device, args.noise_seed,
                        vgg16=ref_vgg16 if sigma == 0 else None, drift_ids=use_drift)
        save_records(out_dir, sigma, run)
        m = sigma_metrics(run.records, args.n_boot)
        if run.vgg16_max_abs_diff is not None:
            m["t0_vs_vgg16_max_abs_logit_diff"] = run.vgg16_max_abs_diff
        metrics[sigma_tag(sigma)] = m
        timing["per_sigma"][sigma_tag(sigma)] = {"seconds": run.seconds, "s_per_image": run.seconds / n}
        if run.drift is not None:
            np.savez_compressed(out_dir / "drift.npz", drift=run.drift, val_id=run.drift_val_ids)
            (out_dir / "drift_summary.json").write_text(json.dumps(
                {"sigma": sigma, "n_images": len(run.drift_val_ids),
                 "definition": "||rep(t) - rep(0)|| / ||rep(0)||, Frobenius per image, then mean/median",
                 **drift_summary(run.drift)}, indent=2) + "\n")
        timing["gpu_memory"] = gpu_memory(device)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        acc = m["top1"]
        print(f"sigma {sigma}: top-1 t=0 {100 * acc[0]:.2f} % -> t={args.T} {100 * acc[-1]:.2f} % "
              f"(gain {100 * m['gain_T']['gain']:+.2f} pp, CI [{100 * m['gain_T']['ci95'][0]:+.2f}, "
              f"{100 * m['gain_T']['ci95'][1]:+.2f}]), {run.seconds / n:.3f} s/image"
              + (f", t0 vs VGG16 max|diff| {run.vgg16_max_abs_diff:.2e}" if run.vgg16_max_abs_diff is not None else ""),
              flush=True)
    print(json.dumps(timing["gpu_memory"]), flush=True)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
