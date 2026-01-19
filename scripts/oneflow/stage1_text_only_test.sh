#!/usr/bin/env bash
set -euo pipefail

# Stage 1 (text-only) test wrapper.
#
# This aligns with doc/oneflow/oneflow_zero_validation_zh.md Stage 1 and runs:
# - Eq(7) loss unit tests
# - sampler-step probability + insertion semantics tests
# - X_t + bag-of-tokens construction tests (subset of sequence_ops)
#
# Usage:
#   bash scripts/oneflow/stage1_text_only_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate Ascend/NPU env (optional; safe on non-NPU boxes).
if [[ -f "${REPO_ROOT}/activate_python_env.sh" ]]; then
  # NOTE: Ascend env scripts may reference unset variables (e.g. ZSH_VERSION),
  # which breaks under `set -u`. Temporarily disable nounset while sourcing.
  set +u
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/activate_python_env.sh"
  set -u
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

pytest -q scripts/tests/test_oneflow_text_loss_eq7.py
pytest -q scripts/tests/test_oneflow_sampler_step.py
pytest -q scripts/tests/test_oneflow_sequence_ops.py -k "build_noised_xt_and_bags"

