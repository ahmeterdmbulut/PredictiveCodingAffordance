"""Write configs/pcoder_c_sqrt.json: PCoderN sqrt(C), calibrated once on CPU (cuDNN off).

Usage (repo root):
  python scripts/calibrate_c_sqrt.py --input imagenet    # official
  python scripts/calibrate_c_sqrt.py --input synthetic   # temporary
"imagenet" uses the 16 images in configs/imagenet_val_c_sqrt_subset.json (val_id 1..16,
first by original filename), extracted to data/imagenet_val/images by
scripts/prepare_imagenet_val.py; "synthetic" uses seeded randn [4, 3, 224, 224] and is
marked temporary. See src/c_sqrt_calibration.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.c_sqrt_calibration import (  # noqa: E402
    N_IMAGES, calibrate, imagenet_files, load_images, make_record, synthetic_images,
)
from src.imagenet_val import C_SQRT_SUBSET_PATH, DEFAULT_IMAGE_DIR  # noqa: E402
from src.load_predify import DEFAULT_C_SQRT_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", choices=["synthetic", "imagenet"], required=True)
    p.add_argument("--imagenet-val-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="extracted subset JPEGs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_C_SQRT_PATH)
    p.add_argument("--force", action="store_true", help="overwrite an existing --out file")
    p.add_argument("--weights-dir", type=Path, default=ROOT / "weights/predify/weights_pvgg16_imagenet")
    p.add_argument("--external-dir", type=Path, default=ROOT / "external")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists; pass --force to overwrite it.")
    files = None
    if args.input == "imagenet":
        files = imagenet_files(C_SQRT_SUBSET_PATH, args.imagenet_val_dir)
        x = load_images(files)                                   # [16, 3, 224, 224]
    else:
        x = synthetic_images(N_IMAGES["synthetic"], args.seed)   # [4, 3, 224, 224]
    values = calibrate(x, args.seed, args.weights_dir, args.external_dir)
    record = make_record(values, args.input, args.seed, files,
                         args.imagenet_val_dir if files else None, args.external_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    for n, (s, c) in enumerate(zip(record["c_sqrt"], record["c"]), start=1):
        print(f"pcoder{n}: C_sqrt = {s:.6f}   C = {c:.2f}")
    print(f"Saved to {args.out} (temporary = {record['temporary']})")


if __name__ == "__main__":
    main()
