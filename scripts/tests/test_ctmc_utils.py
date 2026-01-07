"""
Unit tests for `dllm.pipelines.ctmc_utils`.

Run:
  pytest -q scripts/tests/test_ctmc_utils.py
"""

import torch

from dllm.pipelines.ctmc_utils import (
    bernoulli_from_rate,
    pad_1d,
    safe_log,
    sample_from_logits,
)


def test_pad_1d_shapes_and_mask():
    out, mask = pad_1d([[1, 2, 3], [4], []], pad_val=0)
    assert out.shape == (3, 3)
    assert mask.shape == (3, 3)
    assert out.tolist() == [[1, 2, 3], [4, 0, 0], [0, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0], [0, 0, 0]]


def test_pad_1d_empty_batch():
    out, mask = pad_1d([], pad_val=0)
    assert out.shape == (0, 0)
    assert mask.shape == (0, 0)


def test_sample_from_logits_temperature_zero_is_argmax():
    logits = torch.tensor([0.1, 0.2, -0.3], dtype=torch.float32)
    assert sample_from_logits(logits, temperature=0.0) == 1
    assert sample_from_logits(logits, temperature=-1.0) == 1


def test_sample_from_logits_temperature_positive_in_range():
    torch.manual_seed(0)
    logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    a = sample_from_logits(logits, temperature=1.0)
    assert isinstance(a, int)
    assert 0 <= a < logits.numel()


def test_bernoulli_from_rate_clamps_and_is_binary():
    # This should NOT raise even though rate*tau > 1 for the last element,
    # because bernoulli_from_rate clamps probabilities into [0, 1).
    rate = torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float32)
    out = bernoulli_from_rate(rate, tau=1.0)
    assert out.shape == rate.shape

    # Negative/zero rates clamp to p=0 -> always 0
    assert float(out[0].item()) == 0.0
    assert float(out[1].item()) == 0.0

    # Samples must be binary (0/1)
    uniq = set(float(x) for x in out.detach().cpu().tolist())
    assert uniq.issubset({0.0, 1.0})

    # tau=0 -> p=0 -> always 0
    out0 = bernoulli_from_rate(torch.tensor([100.0, 0.5]), tau=0.0)
    assert torch.all(out0 == 0)


def test_safe_log_is_finite_for_zeros():
    x = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y = safe_log(x)
    assert torch.isfinite(y).all()
    assert float(y[1].item()) == 0.0


def test_safe_log_is_finite_for_negative_values():
    x = torch.tensor([-1.0, 0.0], dtype=torch.float32)
    y = safe_log(x)
    assert torch.isfinite(y).all()


