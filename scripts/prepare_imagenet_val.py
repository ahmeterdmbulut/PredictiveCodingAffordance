"""Write the fixed ImageNet val subsets to configs/ and extract their JPEGs.

Usage (repo root, after the `hf download` in README):
  python scripts/prepare_imagenet_val.py            # refuses to overwrite existing subset files
  python scripts/prepare_imagenet_val.py --force

(a) configs/imagenet_val_c_sqrt_subset.json: 16 images, val_id 1..16 (first by filename).
(b) configs/imagenet_val_gate1_subset.json: 2 per class (2,000), numpy default_rng(seed),
    calibration images excluded. JPEGs go to data/imagenet_val/images/ (git-ignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imagenet_val import (  # noqa: E402
    C_SQRT_SUBSET_PATH, DEFAULT_IMAGE_DIR, DEFAULT_PARQUET_DIR, GATE1_SUBSET_PATH,
    build_index, check_index, extract_jpegs, save_subset, select_calibration, select_per_class,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--n-calibration", type=int, default=16)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite existing subset files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    existing = [p for p in (C_SQRT_SUBSET_PATH, GATE1_SUBSET_PATH) if p.exists()]
    if existing and not args.force:
        sys.exit(f"{[p.name for p in existing]} exist; pass --force to overwrite.")
    index = build_index(args.parquet_dir)
    check_index(index)
    print(f"index: {len(index)} images, {len({e.label for e in index})} classes")

    calib = select_calibration(index, args.n_calibration)
    gate1 = select_per_class(index, args.per_class, args.seed, exclude={e.val_id for e in calib})
    save_subset(C_SQRT_SUBSET_PATH, calib,
                rule=f"first {args.n_calibration} val images by original filename (val_id 1..{args.n_calibration})")
    save_subset(GATE1_SUBSET_PATH, gate1, extra={"seed": args.seed, "per_class": args.per_class},
                rule=(f"{args.per_class} per class: classes 0..999 in order, candidates sorted by val_id, "
                      f"c_sqrt calibration images excluded, numpy.random.default_rng(seed).choice(n, "
                      f"{args.per_class}, replace=False)"))
    print(f"saved {C_SQRT_SUBSET_PATH.name} ({len(calib)}) and {GATE1_SUBSET_PATH.name} ({len(gate1)})")

    written = extract_jpegs(calib + gate1, args.parquet_dir, args.image_dir)
    print(f"extracted {written} JPEGs to {args.image_dir}")


if __name__ == "__main__":
    main()
