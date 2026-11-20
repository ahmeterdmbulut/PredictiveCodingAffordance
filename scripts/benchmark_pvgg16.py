"""Throughput and peak memory of PVGG16 PC timesteps (budgeting Gate 1 and affordance runs).

Usage (repo root):  python scripts/benchmark_pvgg16.py --devices cpu mps --batch-sizes 1 8 32
Writes results/benchmark_<platform>_<timestamp>/{config.json, benchmark.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load_predify import build_pvgg16, c_sqrt_values, get_device, iterate_timesteps  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", nargs="+", default=["cpu", "mps", "cuda"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--pcoder-type", choices=["PCoder", "PCoderN"], default="PCoderN")
    p.add_argument("--hp-set", choices=["fixed", "tuned"], default="fixed")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights/predify/weights_pvgg16_imagenet")
    p.add_argument("--external-dir", type=Path, default=ROOT / "external")
    p.add_argument("--out-root", type=Path, default=ROOT / "results")
    return p.parse_args()


def available(name: str) -> bool:
    return {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_now(device: torch.device) -> float:
    """Bytes in use: CUDA allocator, MPS driver, or process RSS for CPU."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return float(torch.mps.driver_allocated_memory())
    return float(psutil.Process().memory_info().rss)


SWAP_GROWTH_LIMIT = 1e9  # bytes; more swap than this during a config = "does not fit"


class DoesNotFit(Exception):
    pass


def check_swap(device: torch.device, swap_start: float) -> None:
    """On CPU/MPS (shared RAM), abort once the system starts swapping: timings would be meaningless."""
    if device.type != "cuda" and psutil.swap_memory().used - swap_start > SWAP_GROWTH_LIMIT:
        raise DoesNotFit("swapping")


def timed_run(net: torch.nn.Module, x: torch.Tensor, T: int, device: torch.device,
              swap_start: float) -> tuple[float, list[float], float]:
    """One t = 0..T run -> (t0 seconds, per-PC-step seconds, peak memory bytes sampled per step)."""
    step_times, peak = [], memory_now(device)
    sync(device)
    start = time.perf_counter()
    for _ in iterate_timesteps(net, x, T):
        sync(device)
        now = time.perf_counter()
        step_times.append(now - start)
        start = now
        peak = max(peak, memory_now(device))
        check_swap(device, swap_start)
    return step_times[0], step_times[1:], peak


def bench_one(net: torch.nn.Module, device: torch.device, batch: int, args: argparse.Namespace) -> dict:
    row = {"device": str(device), "batch_size": batch, "T": args.T}
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn(batch, 3, 224, 224, generator=gen).to(device)  # [B, 3, 224, 224]
    swap_start = psutil.swap_memory().used
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        timed_run(net, x, 1, device, swap_start)  # warm-up (kernel selection)
        t0s, steps, peaks = [], [], []
        for _ in range(args.repeats):
            t0, pc, peak = timed_run(net, x, args.T, device, swap_start)
            t0s.append(t0)
            steps.extend(pc)
            peaks.append(peak)
    except DoesNotFit as err:
        return {**row, "status": f"does not fit ({err})"}
    except (torch.OutOfMemoryError, RuntimeError) as err:
        if "out of memory" not in str(err).lower() and not isinstance(err, torch.OutOfMemoryError):
            raise
        return {**row, "status": "does not fit (OOM)"}
    t0_img = sum(t0s) / len(t0s) / batch
    step_img = sum(steps) / len(steps) / batch
    return {
        **row, "status": "ok",
        "t0_s_per_image": round(t0_img, 5),
        "pc_step_s_per_image": round(step_img, 5),
        "total_s_per_image_T": round(t0_img + args.T * step_img, 4),
        "pc_step_over_t0": round(step_img / t0_img, 2),
        "peak_mem_gb": round(max(peaks) / 1e9, 2),
        "mem_kind": {"cuda": "max_allocated", "mps": "driver_allocated", "cpu": "process_rss"}[device.type],
    }


CSV_FIELDS = ["device", "batch_size", "T", "status", "t0_s_per_image", "pc_step_s_per_image",
              "total_s_per_image_T", "pc_step_over_t0", "peak_mem_gb", "mem_kind"]


def print_table(rows: list[dict]) -> None:
    print("\n" + " | ".join(CSV_FIELDS))
    for r in rows:
        print(" | ".join(str(r.get(c, "-")) for c in CSV_FIELDS))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / f"benchmark_{platform.system()}-{platform.machine()}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    rows, c_sqrt = [], {}
    for name in args.devices:
        if not available(name):
            print(f"skip {name}: not available")
            continue
        device = get_device(name)
        net = build_pvgg16(weights_dir=args.weights_dir, external_dir=args.external_dir,
                           pcoder_type=args.pcoder_type, hp_set=args.hp_set, device=device)
        fits = True
        for batch in sorted(args.batch_sizes):
            if fits:
                row = bench_one(net, device, batch, args)
                fits = row["status"] == "ok"
            else:  # a smaller batch already did not fit
                row = {"device": str(device), "batch_size": batch, "T": args.T, "status": "skipped (smaller batch did not fit)"}
            print(row, flush=True)
            rows.append(row)
            if device.type == "mps":
                torch.mps.empty_cache()
        c_sqrt[name] = c_sqrt_values(net)
        del net
        if device.type == "mps":
            torch.mps.empty_cache()

    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "platform": platform.platform(), "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "c_sqrt": c_sqrt,
        "notes": ("times exclude warm-up; pc_step = mean over t=1..T; MPS/CPU memory sampled after each step; "
                  "t0 also computes each PCoder's prd and grd (predify/modules/base.py L124-130), "
                  "so it costs about one PC step, not one plain VGG16 pass"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    with (out_dir / "benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print_table(rows)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
