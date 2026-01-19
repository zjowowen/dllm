#!/usr/bin/env bash
set -euo pipefail

# One-click wrapper for PT offline export (save_to_disk).
# It calls: scripts/oneflow/prepare_pt_text_dataset.py
#
# Example (web):
#   bash scripts/oneflow/prepare_pt_bundle.sh \
#     --dataset_name_or_path "HuggingFaceFW/fineweb-edu" \
#     --dataset_config_name "<optional_config_name>" \
#     --text_field text \
#     --tokenizer_name_or_path gpt2 \
#     --seq_length 1024 \
#     --streaming True \
#     --train_split train --test_split None \
#     --output_dir /path/to/bundle_web_1024 \
#     --num_proc 32
#
# Example (code):
#   bash scripts/oneflow/prepare_pt_bundle.sh \
#     --dataset_name_or_path "OpenCoder-LLM/opc-fineweb-code-corpus" \
#     --text_field text \
#     --tokenizer_name_or_path gpt2 \
#     --seq_length 1024 \
#     --streaming True \
#     --train_split train --test_split None \
#     --output_dir /path/to/bundle_code_1024

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# Ascend env scripts may reference unset vars (e.g. ZSH_VERSION) and will fail under `set -u`.
# Temporarily disable nounset for environment activation.
set +u
source activate_python_env.sh
set -u

DATASET_NAME_OR_PATH="Trelis/tiny-shakespeare"
DATASET_CONFIG_NAME=""
TRAIN_SPLIT="train"
TEST_SPLIT="test"
TEXT_FIELD="Text"
TOKENIZER_NAME_OR_PATH="gpt2"
SEQ_LENGTH="1024"
STREAMING="False"
TRAIN_LIMIT=""
TEST_LIMIT=""
NUM_PROC="8"
TOKENIZER_BATCH_SIZE="256"
OUTPUT_DIR="/tmp/oneflow_pt_text_bundle"
KEEP_LABELS="False"
INSERT_EOS="True"
DROP_TAIL="True"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/prepare_pt_bundle.sh [options]

Options:
  --dataset_name_or_path <str>     HF dataset name or local path
  --dataset_config_name <str>      Optional HF config name
  --train_split <str>              Default: train
  --test_split <str>               Default: test (use "None" to disable)
  --text_field <str>               Default: Text
  --tokenizer_name_or_path <str>   Default: gpt2
  --seq_length <int>               Default: 1024
  --streaming <True|False>         Default: False
  --train_limit <int>              Optional, raw-row limit (streaming recommended)
  --test_limit <int>               Optional, raw-row limit
  --num_proc <int>                 Default: 8
  --tokenizer_batch_size <int>     Default: 256 (streaming mode)
  --output_dir <path>              Default: /tmp/oneflow_pt_text_bundle
  --keep_labels <True|False>       Default: False
  --insert_eos <True|False>        Default: True
  --drop_tail <True|False>         Default: True
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_name_or_path) DATASET_NAME_OR_PATH="$2"; shift 2 ;;
    --dataset_config_name) DATASET_CONFIG_NAME="$2"; shift 2 ;;
    --train_split) TRAIN_SPLIT="$2"; shift 2 ;;
    --test_split) TEST_SPLIT="$2"; shift 2 ;;
    --text_field) TEXT_FIELD="$2"; shift 2 ;;
    --tokenizer_name_or_path) TOKENIZER_NAME_OR_PATH="$2"; shift 2 ;;
    --seq_length) SEQ_LENGTH="$2"; shift 2 ;;
    --streaming) STREAMING="$2"; shift 2 ;;
    --train_limit) TRAIN_LIMIT="$2"; shift 2 ;;
    --test_limit) TEST_LIMIT="$2"; shift 2 ;;
    --num_proc) NUM_PROC="$2"; shift 2 ;;
    --tokenizer_batch_size) TOKENIZER_BATCH_SIZE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --keep_labels) KEEP_LABELS="$2"; shift 2 ;;
    --insert_eos) INSERT_EOS="$2"; shift 2 ;;
    --drop_tail) DROP_TAIL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

cmd=(python -u scripts/oneflow/prepare_pt_text_dataset.py
  --dataset_name_or_path "$DATASET_NAME_OR_PATH"
  --train_split "$TRAIN_SPLIT"
  --test_split "$TEST_SPLIT"
  --text_field "$TEXT_FIELD"
  --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
  --seq_length "$SEQ_LENGTH"
  --streaming "$STREAMING"
  --tokenizer_batch_size "$TOKENIZER_BATCH_SIZE"
  --num_proc "$NUM_PROC"
  --output_dir "$OUTPUT_DIR"
  --keep_labels "$KEEP_LABELS"
  --insert_eos "$INSERT_EOS"
  --drop_tail "$DROP_TAIL"
)

if [[ -n "$DATASET_CONFIG_NAME" ]]; then
  cmd+=(--dataset_config_name "$DATASET_CONFIG_NAME")
fi
if [[ -n "$TRAIN_LIMIT" ]]; then
  cmd+=(--train_limit "$TRAIN_LIMIT")
fi
if [[ -n "$TEST_LIMIT" ]]; then
  cmd+=(--test_limit "$TEST_LIMIT")
fi

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

TOKENIZERS_PARALLELISM=false "${cmd[@]}"

