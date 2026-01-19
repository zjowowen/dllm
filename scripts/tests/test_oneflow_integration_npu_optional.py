"""
Optional NPU integration tests.

These tests are skipped unless RUN_NPU_TESTS=1 is set.

Run (on an NPU box with proper Ascend env):
  RUN_NPU_TESTS=1 pytest -q scripts/tests/test_oneflow_integration_npu_optional.py
"""

import os

import pytest


if os.environ.get("RUN_NPU_TESTS", "0").lower() not in ("1", "true", "yes"):
    pytest.skip("RUN_NPU_TESTS is not enabled", allow_module_level=True)


def test_npu_tensor_roundtrip():
    import torch

    # This will raise if NPU backend is not properly installed/configured.
    x = torch.randn((4, 4), dtype=torch.float32, device="npu")
    y = (x + 1.0).sum().to("cpu")
    assert torch.isfinite(y).all()


