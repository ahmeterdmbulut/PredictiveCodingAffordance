"""Calibrate PCoderN's sqrt(C) once on CPU and describe how it was done.

Upstream (predify/modules/base.py L185-220) estimates C on the first forward: it
sets the central cell of each PCoder rep to a random value and counts the output
cells of the feedback pmodule that change (exact !=), averaged over 10 repeats.
- On CUDA, cuDNN float noise makes far more cells "change" (pc1 ~340 instead of
  sqrt(75) = 8.66), and the count differs run to run.
- pc1's pmodule is a bare 64->3 5x5 deconv, so C = 3*5*5 = 75 for any input.
  pc2-pc5 end in a ReLU (pvgg16_shared.py L22-46), so their C depends on the input.
We therefore calibrate once on CPU (fixed seed, cuDNN off) and the loader uses
these fixed values on every device (src/load_predify.py).
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from src.imagenet_val import C_SQRT_SUBSET_PATH, load_subset
from src.load_predify import ROOT, build_pvgg16, c_sqrt_values, predify_preprocess

CalibrationInput = Literal["synthetic", "imagenet"]
N_IMAGES: dict[str, int] = {"synthetic": 4, "imagenet": 16}


def synthetic_images(n: int, seed: int) -> torch.Tensor:
    """[n, 3, 224, 224] seeded N(0, 1) in normalized-image space (same as the tests)."""
    return torch.randn(n, 3, 224, 224, generator=torch.Generator().manual_seed(seed))


def imagenet_files(subset_path: Path, image_dir: Path) -> list[Path]:
    """The calibration images listed in subset_path (configs/imagenet_val_c_sqrt_subset.json:
    val_id 1..16, first by original filename), as extracted JPEGs in image_dir."""
    files = [image_dir / e.filename for e in load_subset(subset_path)]
    assert len(files) == N_IMAGES["imagenet"], f"{subset_path.name} lists {len(files)} images"
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing in {image_dir} (run "
                                f"scripts/prepare_imagenet_val.py), e.g. {missing[0]}")
    return files


def load_images(paths: list[Path]) -> torch.Tensor:
    """Predify preprocessing (Resize 224, CenterCrop 224, Normalize) -> [N, 3, 224, 224]."""
    preprocess = predify_preprocess()
    return torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])


def calibrate(x: torch.Tensor, seed: int, weights_dir: Path, external_dir: Path) -> list[float]:
    """Run upstream's C calibration once on CPU with cuDNN off; return sqrt(C) per PCoder.

    x: [N, 3, 224, 224] on CPU. The seed is set right before the forward because
    the perturbation values come from the global RNG (base.py L206).
    """
    net = build_pvgg16(weights_dir=weights_dir, external_dir=external_dir, pcoder_type="PCoderN",
                       hp_set="fixed", device=torch.device("cpu"), c_sqrt_path=None)
    with torch.backends.cudnn.flags(enabled=False):
        torch.manual_seed(seed)
        net(x)  # t = 0: sets each rep, then calibrates C (base.py L251-252)
    values = c_sqrt_values(net)
    assert values is not None
    return values


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root (posix) if inside it, else absolute."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def make_record(values: list[float], input_kind: CalibrationInput, seed: int,
                files: list[Path] | None, val_dir: Path | None, external_dir: Path) -> dict:
    """JSON-serializable record for configs/pcoder_c_sqrt.json."""
    meta: dict = {
        "input": input_kind, "n_images": N_IMAGES[input_kind], "seed": seed,
        "device": "cpu", "cudnn_enabled": False, "torch": torch.__version__,
        "upstream": {sub: _git_head(external_dir / sub) for sub in ("predify", "predify2021")},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if input_kind == "imagenet":
        assert files is not None and val_dir is not None
        meta.update(subset=_repo_relative(C_SQRT_SUBSET_PATH),
                    val_dir=_repo_relative(val_dir), files=[p.name for p in files],
                    preprocessing="Resize(224) + CenterCrop(224) + ImageNet Normalize")
    else:
        meta["preprocessing"] = "none: seeded randn in normalized-image space"
    return {
        "temporary": input_kind != "imagenet",
        "note": ("Temporary: synthetic input, only so tests can run. Regenerate with --input imagenet "
                 "before Gate 1." if input_kind != "imagenet" else "Official values for all experiments."),
        "c_sqrt": values,
        "c": [round(v * v, 3) for v in values],  # C_sqrt**2, for reading only; the loader uses c_sqrt
        "metadata": meta,
    }


def inputs_from_record(record: dict) -> torch.Tensor:
    """Rebuild the calibration input a record describes. Raises FileNotFoundError if data is missing."""
    meta = record["metadata"]
    if meta["input"] == "synthetic":
        return synthetic_images(meta["n_images"], meta["seed"])
    val_dir = Path(meta["val_dir"])
    val_dir = val_dir if val_dir.is_absolute() else ROOT / val_dir
    files = [val_dir / name for name in meta["files"]]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} calibration images missing in {val_dir}, e.g. {missing[0]}")
    return load_images(files)
