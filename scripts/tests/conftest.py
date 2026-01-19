import os

# Many CPU-only environments have `torch_npu` installed but missing its runtime libs.
# Disable backend auto-loading unless NPU tests are explicitly requested.
if os.environ.get("RUN_NPU_TESTS", "0").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Keep unit tests offline by default.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


