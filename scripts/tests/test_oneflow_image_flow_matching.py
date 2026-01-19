"""
Unit tests for OneFlow image flow matching loss (Eq. 9).

Run:
  pytest -q scripts/tests/test_oneflow_image_flow_matching.py
"""

import torch

from dllm.pipelines.oneflow.losses import image_loss_flow_matching


def test_image_flow_matching_no_modality_is_zero():
    v = torch.zeros((2, 3, 4), dtype=torch.float32)
    flow = torch.zeros((2, 3, 4), dtype=torch.float32)
    is_mod = torch.zeros((2, 3), dtype=torch.bool)
    out = image_loss_flow_matching(v=v, flow_tgt=flow, is_any_modality=is_mod, normalize_by_tokens=True)
    assert float(out.tokens_total.item()) == 0.0
    assert float(out.loss.item()) == 0.0


def test_image_flow_matching_oracle_v_equals_flow_zero_loss():
    v = torch.randn((1, 5, 4), dtype=torch.float32)
    flow = v.clone()
    is_mod = torch.tensor([[False, True, True, False, True]])
    out = image_loss_flow_matching(v=v, flow_tgt=flow, is_any_modality=is_mod, normalize_by_tokens=True)
    assert float(out.loss.item()) == 0.0
    assert float(out.tokens_total.item()) == 3.0


def test_image_flow_matching_known_value():
    # One modality token, difference of 2.0 on each of 4 dims => sq sum = 16
    v = torch.zeros((1, 2, 4), dtype=torch.float32)
    flow = torch.zeros((1, 2, 4), dtype=torch.float32)
    is_mod = torch.tensor([[False, True]])
    flow[0, 1] = 2.0
    out = image_loss_flow_matching(v=v, flow_tgt=flow, is_any_modality=is_mod, normalize_by_tokens=True)
    assert float(out.tokens_total.item()) == 1.0
    assert abs(float(out.loss.item()) - 16.0) < 1e-6


