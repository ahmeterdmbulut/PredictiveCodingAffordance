"""Tests for src/load_predify.py. Run with: python -m pytest tests/test_load_predify.py -v -s"""
from __future__ import annotations

import gc
import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.models import VGG16_Weights, vgg16

from src.c_sqrt_calibration import calibrate, inputs_from_record
from src.load_predify import (
    DEFAULT_C_SQRT_PATH, HP_SETS, N_PCODERS, REP_SHAPES, build_pvgg16, c_sqrt_values,
    iterate_timesteps, load_c_sqrt_file, predify_preprocess, run_timesteps,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "external"
WEIGHTS_DIR = ROOT / "weights" / "predify" / "weights_pvgg16_imagenet"
SANITY_DIR = ROOT / "data" / "sanity_images"
ATOL = 1e-5

pytestmark = pytest.mark.skipif(
    not (WEIGHTS_DIR / "pvgg16_imagenet_pretrained_pc1_pmodule.pth").exists()
    or not (EXTERNAL_DIR / "predify2021").exists(),
    reason="Needs weights/ and external/ (see README 'Upstream versions').",
)

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
]
PCODER_TYPES = ["PCoder", "PCoderN"]

torch.backends.cudnn.benchmark = False  # same conv kernels for both models on CUDA


@lru_cache(maxsize=None)
def _net(device: str, pcoder_type: str, hp_set: str = "fixed") -> torch.nn.Module:
    torch.manual_seed(0)  # deterministic construction (random init before weights load)
    return build_pvgg16(weights_dir=WEIGHTS_DIR, external_dir=EXTERNAL_DIR,
                        pcoder_type=pcoder_type, hp_set=hp_set, device=torch.device(device))


@lru_cache(maxsize=None)
def _reference_vgg(device: str) -> torch.nn.Module:
    return vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval().to(device)


def _free_cached_models() -> None:
    _net.cache_clear()
    _reference_vgg.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _synthetic_images(n: int, device: str) -> torch.Tensor:
    """[n, 3, 224, 224] seeded N(0, 1) inputs in normalized-image space."""
    gen = torch.Generator().manual_seed(0)
    return torch.randn(n, 3, 224, 224, generator=gen).to(device)


@pytest.mark.parametrize("pcoder_type", PCODER_TYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_t0_equals_plain_vgg16(device: str, pcoder_type: str) -> None:
    x = _synthetic_images(4, device)                             # [4, 3, 224, 224]
    with torch.no_grad():
        expected = _reference_vgg(device)(x)                     # [4, 1000]
    logits = next(iterate_timesteps(_net(device, pcoder_type), x, T=0))[1]
    max_diff = (logits - expected).abs().max().item()
    print(f"\n[t=0 synthetic] {device} {pcoder_type}: max|diff| = {max_diff:.3e}")
    assert torch.allclose(logits, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("hp_set", ["fixed", "tuned"])
@pytest.mark.parametrize("pcoder_type", PCODER_TYPES)
def test_config(pcoder_type: str, hp_set: str) -> None:
    net = _net("cpu", pcoder_type, hp_set)
    hp = HP_SETS[hp_set]
    for n in range(1, N_PCODERS + 1):
        assert type(getattr(net, f"pcoder{n}")).__name__ == pcoder_type
        for key in ("ffm", "fbm", "erm"):
            buf = getattr(net, f"{key}{n}")
            assert buf.dtype == torch.float32
            assert buf.item() == pytest.approx(hp[key][n - 1])
    assert net.random_init is False and not net.build_graph
    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())


@pytest.mark.parametrize("pcoder_type", PCODER_TYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_eq2_one_step_pcoder1(device: str, pcoder_type: str) -> None:
    """Hand check of Eq. 2 for pcoder1, t = 0 -> 1 (modules/base.py L120 / L247).

    The input is constant, so ff1(1) = relu1_2(x) = rep1(0), and
    rep1(1) = beta*ff1(1) + lambda*prd2(0) + (1-beta-lambda)*rep1(0) - alpha*s*grd1(0),
    with s = 1 (PCoder) or numel(prd1)/C_sqrt1 (PCoderN).
    """
    net = _net(device, pcoder_type)
    steps = iterate_timesteps(net, _synthetic_images(2, device), T=1)
    next(steps)
    rep1_0 = net.pcoder1.rep.clone()        # [2, 64, 224, 224]
    prd2_0 = net.pcoder2.prd.clone()        # [2, 64, 224, 224] feedback from pcoder2
    grd1_0 = net.pcoder1.grd.clone()        # [2, 64, 224, 224] d eps0 / d rep1
    scale = net.pcoder1.prd.numel() / net.pcoder1.C_sqrt if pcoder_type == "PCoderN" else 1.0
    next(steps)
    beta, lam, alpha = net.ffm1, net.fbm1, net.erm1
    expected = beta * rep1_0 + lam * prd2_0 + (1 - beta - lam) * rep1_0 - alpha * scale * grd1_0
    max_diff = (net.pcoder1.rep - expected).abs().max().item()
    print(f"\n[Eq.2 pcoder1] {device} {pcoder_type}: max|diff| = {max_diff:.3e}")
    assert torch.allclose(net.pcoder1.rep, expected, atol=ATOL, rtol=1e-5)


@pytest.mark.parametrize("pcoder_type", PCODER_TYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_run_timesteps_smoke(device: str, pcoder_type: str) -> None:
    net = _net(device, pcoder_type)
    out = run_timesteps(net, _synthetic_images(4, device), T=2, return_reps=True)
    assert out.logits.shape == (3, 4, 1000) and torch.isfinite(out.logits).all()
    assert out.reps is not None and len(out.reps) == 3
    for step in out.reps:
        assert [tuple(r.shape) for r in step] == [(4, *s) for s in REP_SHAPES]
    assert not torch.allclose(out.logits[1], out.logits[0], atol=1e-4), "PC dynamics had no effect"
    c_sqrt = c_sqrt_values(net)
    print(f"\n[C_sqrt] {device} {pcoder_type}: {c_sqrt}")
    if pcoder_type == "PCoderN":
        assert c_sqrt == load_c_sqrt_file()
    else:
        assert c_sqrt is None


@pytest.mark.parametrize("device", DEVICES)
def test_c_sqrt_fixed(device: str) -> None:
    """Loaded sqrt(C) equals the configs file and is not recalibrated by running PC steps."""
    expected = load_c_sqrt_file()
    net = _net(device, "PCoderN")
    assert c_sqrt_values(net) == expected
    run_timesteps(net, _synthetic_images(2, device), T=2)
    assert c_sqrt_values(net) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_c_sqrt_cpu_equals_cuda() -> None:
    cpu, cuda = c_sqrt_values(_net("cpu", "PCoderN")), c_sqrt_values(_net("cuda", "PCoderN"))
    print(f"\n[C_sqrt] cpu  {cpu}\n[C_sqrt] cuda {cuda}")
    assert cpu == cuda


def test_c_sqrt_pc1_is_sqrt75() -> None:
    """pcoder1's pmodule is a bare 64->3 5x5 deconv: one input cell changes 3*5*5 = 75 outputs."""
    assert c_sqrt_values(_net("cpu", "PCoderN"))[0] == torch.tensor(75.0).sqrt().item()


def test_c_sqrt_file_reproducible() -> None:
    """Recalibrating on CPU with the input the file records gives the same values exactly."""
    record = json.loads(DEFAULT_C_SQRT_PATH.read_text())
    try:
        x = inputs_from_record(record)
    except FileNotFoundError as err:
        pytest.skip(f"Cannot rebuild the {record['metadata']['input']} calibration input: {err}")
    _free_cached_models()  # one more PVGG16 on top of the cache exhausts Windows commit memory
    values = calibrate(x, record["metadata"]["seed"], WEIGHTS_DIR, EXTERNAL_DIR)
    print(f"\n[C_sqrt recalibrated] {values}")
    assert values == record["c_sqrt"]


def _sanity_batch() -> tuple[list[str], torch.Tensor]:
    paths = sorted(SANITY_DIR.glob("*.jpg"))
    if not paths:
        pytest.skip(f"No JPEGs in {SANITY_DIR} (see README).")
    preprocess = predify_preprocess()
    x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])  # [N, 3, 224, 224]
    return [p.name for p in paths], x


@pytest.mark.parametrize("pcoder_type", PCODER_TYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_t0_real_images_top5(device: str, pcoder_type: str) -> None:
    names, x = _sanity_batch()
    x = x.to(device)
    with torch.no_grad():
        expected = _reference_vgg(device)(x)                     # [N, 1000]
    logits = next(iterate_timesteps(_net(device, pcoder_type), x, T=0))[1]
    max_diff = (logits - expected).abs().max().item()
    print(f"\n[t=0 real] {device} {pcoder_type}: max|diff| = {max_diff:.3e}")
    assert torch.allclose(logits, expected, atol=ATOL, rtol=0)

    categories = VGG16_Weights.IMAGENET1K_V1.meta["categories"]
    top5 = logits.softmax(-1).topk(5, dim=-1)
    assert torch.equal(top5.indices, expected.topk(5, dim=-1).indices)
    for name, probs, idx in zip(names, top5.values.cpu(), top5.indices.cpu()):
        labels = ", ".join(f"{categories[i]} {p:.2f}" for p, i in zip(probs.tolist(), idx.tolist()))
        print(f"  {name:12s} {labels}")
