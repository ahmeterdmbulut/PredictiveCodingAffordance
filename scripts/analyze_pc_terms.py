"""Size of the alpha (error) term vs the other Predify update terms, per PCoder, on a few images.

Diagnostic for the Gate 1 observation that PCoder and PCoderN give nearly the same accuracy.
Uses the first --n-images Gate 1 images as ONE batch (plain PCoder's step depends on batch
size; 16 matches the Gate 1 runs, whose first batch is exactly these images), the same
seeded noise, and all four configurations. No accuracy is computed.

Usage (repo root):  python scripts/analyze_pc_terms.py [--device cuda] [--n-images 16]
Writes results/gate1_pc_terms_<timestamp>/{config.json, terms.json, terms.md}.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degrade import add_gaussian_noise, seeded_gaussian_noise  # noqa: E402
from src.imagenet_val import GATE1_SUBSET_PATH, ImageNetSubset  # noqa: E402
from src.load_predify import N_PCODERS, build_pvgg16, get_device, iterate_timesteps  # noqa: E402
from src.pc_terms import TERMS, record_update_terms  # noqa: E402

CONFIGS = [("fixed", "PCoderN"), ("tuned", "PCoderN"), ("fixed", "PCoder"), ("tuned", "PCoder")]
REPORT_T = (1, 5, 15)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None)
    p.add_argument("--n-images", type=int, default=16)
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights/predify/weights_pvgg16_imagenet")
    p.add_argument("--external-dir", type=Path, default=ROOT / "external")
    p.add_argument("--out-root", type=Path, default=ROOT / "results")
    return p.parse_args()


def summarize(log: list[list[dict[str, torch.Tensor]]]) -> dict:
    """Mean over images of each term norm and of ||err|| / ||ff term||, per PCoder and t = 1..T."""
    out = {}
    for k, steps in enumerate(log):
        out[f"pcoder{k + 1}"] = {
            **{name: [s[name].mean().item() for s in steps] for name in TERMS},
            "err_over_ff": [(s["err"] / s["ff"]).mean().item() for s in steps],
        }
    return out


def markdown(results: dict) -> str:
    lines = ["# Predify update terms per PCoder (mean over images)", "",
             "Ratio ||alpha * scale * grad|| / ||beta * ff|| (per image, then mean). "
             "Columns: t = " + ", ".join(map(str, REPORT_T)) + ".", ""]
    for sigma, by_cfg in results.items():
        lines += [f"## sigma = {sigma}", "",
                  "| config | " + " | ".join(f"pc{k}" for k in range(1, N_PCODERS + 1)) + " |",
                  "|---|" + "---|" * N_PCODERS]
        for cfg, by_pc in by_cfg.items():
            cells = [" / ".join(f"{by_pc[f'pcoder{k}']['err_over_ff'][t - 1]:.2e}" for t in REPORT_T)
                     for k in range(1, N_PCODERS + 1)]
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines += ["", "Absolute term norms at t = 1 (ff / fb / mem / err):", "",
                  "| config | " + " | ".join(f"pc{k}" for k in range(1, N_PCODERS + 1)) + " |",
                  "|---|" + "---|" * N_PCODERS]
        for cfg, by_pc in by_cfg.items():
            cells = [" / ".join(f"{by_pc[f'pcoder{k}'][n][0]:.3g}" for n in TERMS) for k in range(1, N_PCODERS + 1)]
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = False
    dataset = ImageNetSubset(GATE1_SUBSET_PATH)
    items = [dataset[i] for i in range(args.n_images)]
    x = torch.stack([it[0] for it in items])                     # [B, 3, 224, 224]
    val_ids = [it[2] for it in items]
    eps = seeded_gaussian_noise(val_ids, shape=tuple(x.shape[1:]), seed=args.noise_seed)

    results: dict = {}
    for hp_set, pcoder_type in CONFIGS:
        torch.manual_seed(0)
        net = build_pvgg16(weights_dir=args.weights_dir, external_dir=args.external_dir,
                           pcoder_type=pcoder_type, hp_set=hp_set, device=device)
        for sigma in args.sigmas:
            xs = add_gaussian_noise(x, eps, sigma).to(device)
            with torch.no_grad(), record_update_terms(net) as log:
                for _ in iterate_timesteps(net, xs, args.T):
                    pass
            results.setdefault(f"{sigma:.2f}", {})[f"{hp_set}_{pcoder_type}"] = summarize(log)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = args.out_root / f"gate1_pc_terms_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device": str(device), "val_ids": val_ids, "batch": "all n_images in one batch",
              "terms": "ff = beta*ff, fb = lambda*fb, mem = (1-beta-lambda)*rep(t), "
                       "err = alpha*scale*grad (scale = prd.numel()/C_sqrt for PCoderN, 1 for PCoder); "
                       "norms per image over C,H,W; t = 1..T",
              "torch": torch.__version__, "platform": platform.platform()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "terms.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "terms.md").write_text(markdown(results))
    print(markdown(results))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
