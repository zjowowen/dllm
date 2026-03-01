import os
import sys

# Add scripts/tests to path so that "attention" is a top-level package
# when test files use: from .common import ...
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Many CPU-only environments have `torch_npu` installed but missing its runtime libs.
# Disable backend auto-loading unless NPU tests are explicitly requested.
if os.environ.get("RUN_NPU_TESTS", "0").lower() not in ("1", "true", "yes"):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Keep unit tests offline by default.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


