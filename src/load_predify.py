"""Load the frozen PVGG16 from Predify (Choksi et al., 2021) and run its PC timesteps.

Wraps the pinned upstream code in external/ without editing it:
- external/predify2021/predify2021/model_factory/pvgg16_shared.py  (PVGG16SeparateHP)
- external/predify/predify/modules/base.py                         (PCoder, PCoderN)
Upstream get_model() is broken (undefined `opj`/`net`) and is never called.
PCoderN's sqrt(C) is loaded from configs/pcoder_c_sqrt.json instead of being
calibrated at runtime (see src/c_sqrt_calibration.py for why).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

N_PCODERS = 5
PCoderType = Literal["PCoder", "PCoderN"]
HPSetName = Literal["fixed", "tuned"]

# beta = ffm (feedforward), lambda = fbm (feedback), alpha = erm (error correction), one per PCoder.
HP_SETS: dict[str, dict[str, tuple[float, ...]]] = {
    # Choksi 2021 Sec. 2.2; used for Table S4 / Fig. 2a (App. A.7).
    "fixed": {"ffm": (0.8,) * 5, "fbm": (0.1,) * 5, "erm": (0.01,) * 5},
    # Choksi 2021 App. A.7 Table S2 (tuned on Gaussian noise sigma = 0.5).
    "tuned": {
        "ffm": (0.2, 0.4, 0.4, 0.5, 0.6),
        "fbm": (0.05, 0.10, 0.10, 0.10, 0.00),
        "erm": (0.01,) * 5,
    },
}

# Per-sample shapes of pcoder1..5.rep (VGG16 relu1_2, relu2_2, relu3_3, relu4_3, relu5_3).
REP_SHAPES: tuple[tuple[int, int, int], ...] = (
    (64, 224, 224), (128, 112, 112), (256, 56, 56), (512, 28, 28), (512, 14, 14),
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_SQRT_PATH = ROOT / "configs" / "pcoder_c_sqrt.json"


def get_device(name: str | None = None) -> torch.device:
    """Pick cuda > mps > cpu, or validate an explicit choice such as "cpu" or "cuda:0"."""
    if name is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {name} but CUDA is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Requested {name} but MPS is not available.")
    return device


def predify_preprocess() -> transforms.Compose:
    """PIL image -> normalized tensor [3, 224, 224], as in predify2021 calculate_corr_dists.py L75-79."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _import_upstream(external_dir: Path) -> tuple[type, type, type]:
    """Return (PVGG16SeparateHP, PCoder, PCoderN) from the clones in external_dir."""
    for sub in ("predify", "predify2021"):  # predify may also be pip-installed (-e)
        path = str((external_dir / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    from predify.modules import PCoder, PCoderN
    from predify2021.model_factory.pvgg16_shared import PVGG16SeparateHP
    return PVGG16SeparateHP, PCoder, PCoderN


def _check_hp_set(hp: dict[str, tuple[float, ...]]) -> None:
    for key in ("ffm", "fbm", "erm"):
        assert len(hp[key]) == N_PCODERS, f"{key} needs {N_PCODERS} values"
    for beta, lam in zip(hp["ffm"], hp["fbm"]):
        assert 0.0 <= beta + lam <= 1.0, "Eq. 2 needs 0 <= beta + lambda <= 1"


def _set_pcoder_type(net: nn.Module, cls: type) -> None:
    """Swap every pcoderN for `cls`, keeping its pmodule and feedback flag.

    Safe without editing upstream: the hooks look up self.pcoderN at call time
    (pvgg16_shared.py L17, L25, ...) and net.pcoders is only built on the first
    reset (predify/networks/base.py L284-288).
    """
    assert net.pcoders is None, "Swap PCoder type before the first forward pass."
    for n in range(1, N_PCODERS + 1):
        old = getattr(net, f"pcoder{n}")
        setattr(net, f"pcoder{n}", cls(old.pmodule, old.has_feedback, False))


def _load_feedback_weights(net: nn.Module, weights_dir: Path) -> None:
    """Load the five pmodule state_dicts (feedback deconvolutions) from weights_dir.

    All five files use keys "0.weight"/"0.bias" (saved from a Sequential). pcoder1's
    pmodule is a bare ConvTranspose2d (pvgg16_shared.py L14; paper App. A.1 config,
    no ReLU since it predicts the normalized image), so its "0." prefix is stripped.
    Shapes are still checked by strict loading.
    """
    for n in range(1, N_PCODERS + 1):
        path = weights_dir / f"pvgg16_imagenet_pretrained_pc{n}_pmodule.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        pmodule = getattr(net, f"pcoder{n}").pmodule
        if not isinstance(pmodule, nn.Sequential) and all(k.startswith("0.") for k in state):
            state = {k[len("0."):]: v for k, v in state.items()}
        pmodule.load_state_dict(state, strict=True)


def load_c_sqrt_file(path: Path = DEFAULT_C_SQRT_PATH) -> list[float]:
    """Fixed sqrt(C) for pcoder1..5, written by scripts/calibrate_c_sqrt.py."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python scripts/calibrate_c_sqrt.py` (see README).")
    record = json.loads(path.read_text())
    if record.get("temporary", True):
        warnings.warn(f"{path.name} is temporary ({record['metadata']['input']} input); "
                      "regenerate it with --input imagenet before Gate 1.", stacklevel=2)
    values = [float(v) for v in record["c_sqrt"]]
    assert len(values) == N_PCODERS and all(v > 0 for v in values), f"bad c_sqrt in {path}: {values}"
    return values


def _set_fixed_c_sqrt(net: nn.Module, values: list[float]) -> None:
    """Fill each pcoderN.C_sqrt buffer (float32 scalar, base.py L182). Upstream only
    calibrates while C_sqrt == -1 (base.py L251), so this disables runtime calibration."""
    for n, value in enumerate(values, start=1):
        getattr(net, f"pcoder{n}").C_sqrt.fill_(value)


def check_c_sqrt(net: nn.Module, expected: list[float]) -> None:
    """Assert the net's sqrt(C) values equal `expected` exactly (float32 round trip)."""
    actual = c_sqrt_values(net)
    assert actual == expected, f"C_sqrt {actual} != fixed values {expected}"


def build_pvgg16(
    *,
    weights_dir: Path,
    external_dir: Path,
    pcoder_type: PCoderType,
    hp_set: HPSetName,
    device: torch.device,
    c_sqrt_path: Path | None = DEFAULT_C_SQRT_PATH,
) -> nn.Module:
    """Build the frozen PVGG16 (VGG16 IMAGENET1K_V1 + ImageNet feedback weights) in eval mode.

    For PCoderN, sqrt(C) comes from `c_sqrt_path`. None keeps upstream runtime
    calibration; only src/c_sqrt_calibration.py uses that.
    """
    PVGG16SeparateHP, PCoder, PCoderN = _import_upstream(external_dir)
    hp = HP_SETS[hp_set]
    _check_hp_set(hp)

    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Tuples -> float32 buffers ffm1..5 / fbm1..5 / erm1..5 (predify/networks/base.py L221-229).
    net = PVGG16SeparateHP(
        backbone, build_graph=False, random_init=False,
        ff_multiplier=hp["ffm"], fb_multiplier=hp["fbm"], er_multiplier=hp["erm"],
    )
    _set_pcoder_type(net, {"PCoder": PCoder, "PCoderN": PCoderN}[pcoder_type])
    _load_feedback_weights(net, weights_dir)
    fixed_c_sqrt = None
    if pcoder_type == "PCoderN" and c_sqrt_path is not None:
        fixed_c_sqrt = load_c_sqrt_file(c_sqrt_path)
        _set_fixed_c_sqrt(net, fixed_c_sqrt)

    net.requires_grad_(False)  # frozen; PC still takes gradients w.r.t. activations
    net.eval()
    net = net.to(device)
    if fixed_c_sqrt is not None:
        check_c_sqrt(net, fixed_c_sqrt)
    return net


def iterate_timesteps(
    net: nn.Module, x: torch.Tensor, T: int
) -> Iterator[tuple[int, torch.Tensor, tuple[torch.Tensor, ...]]]:
    """Yield (t, logits [B, 1000], reps) for t = 0..T: net(x) at t = 0, then net(None).

    x: [B, 3, 224, 224] normalized images. reps = pcoder1..5.rep with shapes
    [B, *REP_SHAPES[i]], on the model's device. Upstream assigns new tensors each
    step, so tensors from earlier steps stay valid. No grad context is held
    across yields: PNetSeparateHP.forward applies no_grad itself (networks/base.py L271-273).
    """
    assert not net.build_graph and not net.training
    for t in range(T + 1):
        logits = net(x) if t == 0 else net(None)
        reps = tuple(getattr(net, f"pcoder{n}").rep for n in range(1, N_PCODERS + 1))
        yield t, logits, reps


@dataclass
class TimestepOutput:
    logits: torch.Tensor                                  # [T+1, B, 1000], model device
    reps: list[tuple[torch.Tensor, ...]] | None = None    # per step: 5 tensors on CPU


def run_timesteps(net: nn.Module, x: torch.Tensor, T: int, return_reps: bool = False) -> TimestepOutput:
    """Run t = 0..T and stack logits. return_reps copies every rep to CPU
    (24.5 MB per image per step), so use it only for small batches."""
    logits, reps = [], []
    for _, step_logits, step_reps in iterate_timesteps(net, x, T):
        logits.append(step_logits)
        if return_reps:
            reps.append(tuple(r.detach().cpu() for r in step_reps))
    return TimestepOutput(torch.stack(logits), reps if return_reps else None)


def c_sqrt_values(net: nn.Module) -> list[float] | None:
    """sqrt(C) per PCoderN (fixed at build, or set on the first forward if calibrating),
    or None for plain PCoder."""
    pcoders = [getattr(net, f"pcoder{n}") for n in range(1, N_PCODERS + 1)]
    if not all(hasattr(pc, "C_sqrt") for pc in pcoders):
        return None
    return [float(pc.C_sqrt) for pc in pcoders]
