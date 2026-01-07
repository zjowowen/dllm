"""
Shared utilities for CTMC-style edit/insertion samplers and trainers.

This module is intentionally lightweight and dependency-minimal so it can be reused
by multiple pipelines (e.g., EditFlow, OneFlow) without creating circular imports.

Design constraints:
- Keep dependencies minimal: only `torch` is allowed here.
- Do not import pipeline modules (e.g., `dllm.pipelines.editflow.*`) to avoid cycles.
- Functions should be pure helpers (no global state, no heavy side effects).

Public API:
- `pad_1d`: pad python lists of token ids into `[B, L]` LongTensor + 0/1 mask.
- `bernoulli_from_rate`: CTMC rate -> Bernoulli triggers for a step size `tau`.
- `sample_from_logits`: deterministic argmax when temperature<=0, else categorical sample.
- `safe_log`: numerically stable `log(x)` with lower clamping.

Tests:
- See `scripts/tests/test_ctmc_utils.py` (run: `pytest -q scripts/tests/test_ctmc_utils.py`).
"""

from __future__ import annotations

import torch


def pad_1d(
    batch_lists: list[list[int]], pad_val: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of variable-length integer lists into:
      - out: tensor of shape [B, Lmax] with padding value `pad_val`
      - mask: tensor of shape [B, Lmax] with 1 for real tokens and 0 for padding (int mask)
    """

    B = len(batch_lists)
    Lmax = max((len(x) for x in batch_lists), default=0)
    out = torch.full((B, Lmax), pad_val, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.long)  # 0/1 mask (int)

    for b, x in enumerate(batch_lists):
        if not x:
            continue
        L = len(x)
        out[b, :L] = torch.tensor(x, dtype=torch.long)
        mask[b, :L] = 1  # mark valid positions with 1

    return out, mask


def bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Sample Bernoulli events from a per-position CTMC rate with step size `tau`.

    The probability is clamped into [0, 1) for numerical stability.
    """

    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)


def sample_from_logits(logits_row: torch.Tensor, temperature: float) -> int:
    """
    Sample a token id from a 1D logits tensor.

    - If temperature <= 0, returns argmax.
    - Else samples from a categorical distribution with scaled logits.
    """

    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    return int(
        torch.distributions.Categorical(logits=(logits_row / float(temperature)))
        .sample()
        .item()
    )


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable log(x) with lower clamp."""

    return torch.log(x.clamp_min(float(eps)))


