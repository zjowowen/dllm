#!/usr/bin/env bash
set -euo pipefail

# One-click wrapper for SFT offline export (save_to_disk).
# It calls: scripts/oneflow/prepare_sft_text_dataset.py
#
# Examples:
#   bash scripts/oneflow/prepare_sft_bundle.sh \
#     --dataset_args "HuggingFaceH4/ultrachat_200k[train:200000]" \
#     --tokenizer_name_or_path gpt2 \
#     --max_length 1024 \
#     --output_dir /path/to/bundle_sft_ultrachat_1024
#
#   bash scripts/oneflow/prepare_sft_bundle.sh \
#     --dataset_args "OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python][train:200000]" \
#     --tokenizer_name_or_path gpt2 \
#     --max_length 1024 \
#     --output_dir /path/to/bundle_sft_opc_python_1024

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source activate_python_env.sh

DATASET_ARGS="tatsu-lab/alpaca"
TOKENIZER_NAME_OR_PATH="gpt2"
MAX_LENGTH="1024"
TRUNCATION="right"
MASK_PROMPT_LOSS="True"
NUM_PROC="8"
OUTPUT_DIR="/tmp/oneflow_sft_text_bundle"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/prepare_sft_bundle.sh [options]

Options:
  --dataset_args <str>            SFT dataset spec (supports + and [train:...] limits)
  --tokenizer_name_or_path <str>  Default: gpt2
  --max_length <int>              Default: 1024
  --truncation <right|filter>     Default: right
  --mask_prompt_loss <True|False> Default: True
  --num_proc <int>                Default: 8
  --output_dir <path>             Default: /tmp/oneflow_sft_text_bundle
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_args) DATASET_ARGS="$2"; shift 2 ;;
    --tokenizer_name_or_path) TOKENIZER_NAME_OR_PATH="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --truncation) TRUNCATION="$2"; shift 2 ;;
    --mask_prompt_loss) MASK_PROMPT_LOSS="$2"; shift 2 ;;
    --num_proc) NUM_PROC="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

cmd=(python -u scripts/oneflow/prepare_sft_text_dataset.py
  --dataset_args "$DATASET_ARGS"
  --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
  --max_length "$MAX_LENGTH"
  --truncation "$TRUNCATION"
  --mask_prompt_loss "$MASK_PROMPT_LOSS"
  --num_proc "$NUM_PROC"
  --output_dir "$OUTPUT_DIR"
)

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

TOKENIZERS_PARALLELISM=false "${cmd[@]}"

