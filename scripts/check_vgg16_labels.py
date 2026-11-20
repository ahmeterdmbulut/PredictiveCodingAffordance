"""Check that HF ImageNet val labels use torchvision's VGG16 class order.

Plain torchvision VGG16 (IMAGENET1K_V1) + Predify preprocessing on every 10th image
of the Gate 1 subset (200 images, sorted by val_id). Expect top-1 ~70 %
(Choksi 2021 Table S4, t = 0, sigma = 0: 71.63); ~0.1 % would mean a label mismatch.
Also compares the HF class names with VGG16_Weights.meta["categories"].

Usage (repo root):
  python scripts/check_vgg16_labels.py [--device cuda] [--stride 10] [--batch-size 16]
Writes results/vgg16_label_check_<timestamp>/{config.json, metrics.json}.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import VGG16_Weights, vgg16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imagenet_val import GATE1_SUBSET_PATH, ImageNetSubset, class_names  # noqa: E402
from src.load_predify import get_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--stride", type=int, default=10, help="use every n-th image of the Gate 1 subset")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-root", type=Path, default=ROOT / "results")
    return p.parse_args()


def compare_names(hf: list[str], tv: list[str]) -> list[tuple[int, str, str]]:
    """(index, hf, torchvision) where torchvision's name is not one of HF's comma-separated synonyms."""
    return [(i, h, t) for i, (h, t) in enumerate(zip(hf, tv))
            if t.lower() not in {s.strip().lower() for s in h.split(",")}]


@torch.no_grad()
def evaluate(net: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Top-1 / top-5 counts over the loader."""
    top1 = top5 = n = 0
    for x, y, _ in loader:              # x: [B, 3, 224, 224], y: [B]
        logits = net(x.to(device))      # [B, 1000]
        top = logits.topk(5, dim=1).indices.cpu()  # [B, 5]
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    return {"n_images": n, "top1": top1 / n, "top5": top5 / n}


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dataset = ImageNetSubset(GATE1_SUBSET_PATH)
    subset = Subset(dataset, list(range(0, len(dataset), args.stride)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = VGG16_Weights.IMAGENET1K_V1
    net = vgg16(weights=weights).eval().to(device)
    metrics = evaluate(net, loader, device)
    mismatches = compare_names(class_names(), weights.meta["categories"])
    metrics["name_mismatches"] = [{"index": i, "hf": h, "torchvision": t} for i, h, t in mismatches]

    print(f"VGG16 on {metrics['n_images']} images: top-1 {100 * metrics['top1']:.1f} %, "
          f"top-5 {100 * metrics['top5']:.1f} %")
    print(f"class names: {1000 - len(mismatches)}/1000 match")
    for i, h, t in mismatches:
        print(f"  {i}: hf={h!r} torchvision={t!r}")

    out_dir = args.out_root / f"vgg16_label_check_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    config = {**{k: str(v) for k, v in vars(args).items()}, "device": str(device),
              "subset": GATE1_SUBSET_PATH.relative_to(ROOT).as_posix(),
              "images": [dataset.images[i].val_id for i in subset.indices],
              "weights": "torchvision VGG16_Weights.IMAGENET1K_V1",
              "preprocessing": "Resize(224) + CenterCrop(224) + ImageNet Normalize",
              "torch": torch.__version__, "platform": platform.platform()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
