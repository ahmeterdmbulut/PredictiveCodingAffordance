"""ImageNet validation split (HF ILSVRC/imagenet-1k parquet): index, fixed subsets, JPEG extraction.

The parquet rows are shuffled, so images are identified by the number in their
original filename (ILSVRC2012_val_00027555_n01824575.JPEG -> val_id 27555),
which does not depend on how the Hub sharded the data.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.load_predify import ROOT, predify_preprocess

HF_REPO = "ILSVRC/imagenet-1k"
HF_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
N_VAL, N_CLASSES = 50_000, 1000
DEFAULT_PARQUET_DIR = ROOT / "data/imagenet_val/data"
DEFAULT_IMAGE_DIR = ROOT / "data/imagenet_val/images"
C_SQRT_SUBSET_PATH = ROOT / "configs/imagenet_val_c_sqrt_subset.json"
GATE1_SUBSET_PATH = ROOT / "configs/imagenet_val_gate1_subset.json"
_VAL_ID = re.compile(r"ILSVRC2012_val_(\d{8})")


@dataclass(frozen=True)
class ValImage:
    val_id: int     # number in the original filename, 1..50000
    filename: str   # original filename as stored in image.path
    label: int      # 0..999, torchvision class order
    shard: str      # parquet file name
    row: int        # row inside that shard


def parse_val_id(filename: str) -> int:
    match = _VAL_ID.search(filename)
    if match is None:
        raise ValueError(f"not an ImageNet val filename: {filename}")
    return int(match.group(1))


def build_index(parquet_dir: Path = DEFAULT_PARQUET_DIR) -> list[ValImage]:
    """One entry per image, sorted by val_id. Reads only image.path and label (no pixels)."""
    shards = sorted(parquet_dir.glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no validation-*.parquet in {parquet_dir}")
    index = []
    for shard in shards:
        table = pq.read_table(shard, columns=["image.path", "label"])  # pyarrow names the leaf "path"
        paths, labels = table.column("path").to_pylist(), table.column("label").to_pylist()
        for row, (path, label) in enumerate(zip(paths, labels)):
            index.append(ValImage(parse_val_id(path), path, label, shard.name, row))
    return sorted(index, key=lambda e: e.val_id)


def class_names(parquet_dir: Path = DEFAULT_PARQUET_DIR) -> list[str]:
    """The 1000 ClassLabel names stored in the shards' Hugging Face metadata."""
    shard = next(iter(sorted(parquet_dir.glob("validation-*.parquet"))))
    meta = json.loads(pq.read_schema(shard).metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def check_index(index: list[ValImage]) -> None:
    """Full split: 50,000 unique ids 1..50000, labels 0..999, 50 images per class."""
    assert [e.val_id for e in index] == list(range(1, N_VAL + 1)), "val ids are not exactly 1..50000"
    counts = np.bincount([e.label for e in index], minlength=N_CLASSES)
    assert len(counts) == N_CLASSES and (counts == N_VAL // N_CLASSES).all(), "not 50 images per class"


def select_calibration(index: list[ValImage], n: int = 16) -> list[ValImage]:
    """First n images by original filename (val_id 1..n)."""
    return sorted(index, key=lambda e: e.val_id)[:n]


def select_per_class(index: list[ValImage], per_class: int, seed: int,
                     exclude: set[int] = frozenset()) -> list[ValImage]:
    """per_class images of every label, drawn with one seeded generator; sorted by val_id.

    Classes are visited in ascending label order, candidates sorted by val_id, so the
    result depends only on (index, per_class, seed, exclude).
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[ValImage]] = {}
    for e in sorted(index, key=lambda e: e.val_id):
        if e.val_id not in exclude:
            by_class.setdefault(e.label, []).append(e)
    chosen = []
    for label in sorted(by_class):
        candidates = by_class[label]
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has {len(candidates)} images, need {per_class}")
        picks = rng.choice(len(candidates), size=per_class, replace=False)
        chosen += [candidates[i] for i in sorted(picks)]
    return sorted(chosen, key=lambda e: e.val_id)


def save_subset(path: Path, images: list[ValImage], rule: str, extra: dict | None = None) -> None:
    """Write a subset file: how it was chosen + one record per image."""
    record = {
        "source": {"hf_repo": HF_REPO, "revision": HF_REVISION, "split": "validation"},
        "rule": rule,
        **(extra or {}),
        "n_images": len(images),
        "n_classes": len({e.label for e in images}),
        "created": datetime.now().isoformat(timespec="seconds"),
        "images": [asdict(e) for e in images],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")


def load_subset(path: Path) -> list[ValImage]:
    return [ValImage(**rec) for rec in json.loads(path.read_text())["images"]]


def extract_jpegs(images: list[ValImage], parquet_dir: Path = DEFAULT_PARQUET_DIR,
                  out_dir: Path = DEFAULT_IMAGE_DIR) -> int:
    """Write the original JPEG bytes (no re-encoding) of `images` to out_dir/<filename>.

    Returns the number of files written; existing files are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    todo: dict[str, list[ValImage]] = {}
    for e in images:
        if not (out_dir / e.filename).exists():
            todo.setdefault(e.shard, []).append(e)
    for shard, entries in todo.items():
        table = pq.read_table(parquet_dir / shard, columns=["image"])
        column = table.column("image")
        for e in entries:
            rec = column[e.row].as_py()
            assert rec["path"] == e.filename, f"{shard} row {e.row}: {rec['path']} != {e.filename}"
            (out_dir / e.filename).write_bytes(rec["bytes"])
    return sum(len(v) for v in todo.values())


class ImageNetSubset(Dataset):
    """Extracted subset JPEGs -> (image [3, 224, 224] with Predify preprocessing, label, val_id)."""

    def __init__(self, subset_path: Path, image_dir: Path = DEFAULT_IMAGE_DIR) -> None:
        self.images = load_subset(subset_path)
        self.image_dir = image_dir
        self.preprocess = predify_preprocess()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        e = self.images[i]
        with Image.open(self.image_dir / e.filename) as img:
            x = self.preprocess(img.convert("RGB"))
        return x, e.label, e.val_id
