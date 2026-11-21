"""Measure the size of each term of the Predify update (Choksi 2021 Eq. 2) per PCoder.

Upstream update (predify/modules/base.py L117-120 PCoder, L250-256 PCoderN):
  rep(t+1) = ffm * ff + fbm * fb + (1 - ffm - fbm) * rep(t) - erm * scale * grd(t)
  (without feedback: ffm * ff + (1 - ffm) * rep(t) - erm * scale * grd(t))
  scale = 1 for PCoder, prd.numel() / C_sqrt for PCoderN (prd.numel() includes the batch).
A forward pre-hook reads ff, fb (keyword arguments, pvgg16_shared.py L17, L25, ...)
and the PCoder's state before it updates, so upstream code is not edited.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from src.load_predify import N_PCODERS

TERMS = ("ff", "fb", "mem", "err")


def _norm(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B] Frobenius norm per image."""
    return torch.linalg.vector_norm(x.flatten(1), dim=1)


def update_term_norms(pc: nn.Module, ff: torch.Tensor, fb: torch.Tensor | None,
                      ffm: torch.Tensor, fbm: torch.Tensor, erm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image norms [B] of the four update terms, using pc's state before the update."""
    scale = pc.prd.numel() / pc.C_sqrt if hasattr(pc, "C_sqrt") else 1.0
    if pc.has_feedback:
        ff_t, fb_t, mem_t = ffm * ff, fbm * fb, (1 - ffm - fbm) * pc.rep
    else:
        ff_t, fb_t, mem_t = ffm * ff, torch.zeros_like(ff), (1 - ffm) * pc.rep
    err_t = erm * scale * pc.grd
    return {"ff": _norm(ff_t), "fb": _norm(fb_t), "mem": _norm(mem_t), "err": _norm(err_t)}


@contextmanager
def record_update_terms(net: nn.Module) -> Iterator[list[list[dict[str, torch.Tensor]]]]:
    """Yield log[k] = list over PC steps (t = 1, 2, ...) of term norms for pcoder k+1.

    t = 0 (rep is None: rep is just set to ff) is not logged.
    """
    log: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(N_PCODERS)]
    handles = []
    for k in range(N_PCODERS):
        def hook(pc, args, kwargs, k=k):
            if pc.rep is not None:
                norms = update_term_norms(pc, kwargs["ff"], kwargs["fb"], kwargs["ffm"], kwargs["fbm"], kwargs["erm"])
                log[k].append({name: v.detach().cpu() for name, v in norms.items()})
        handles.append(getattr(net, f"pcoder{k + 1}").register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield log
    finally:
        for h in handles:
            h.remove()
