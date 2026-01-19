#!/usr/bin/env bash
set -euo pipefail

# Stage 1 (text-only) smoke training wrapper.
#
# Usage:
#   bash scripts/oneflow/stage1_text_only_train.sh --max_steps 20 --device cpu

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

python -u scripts/oneflow/stage1_text_only_train.py "$@"

