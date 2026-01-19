#!/usr/bin/env bash
set -euo pipefail

# One-click 4-node (64 NPU) PT launch wrapper.
#
# Required env:
#   MASTER_ADDR : rank0 IP/hostname
#   MASTER_PORT : e.g. 29500
#   NODE_RANK   : 0..3
#
# Example (run on each node, only NODE_RANK changes):
#   export MASTER_ADDR="<rank0-ip>"
#   export MASTER_PORT=29500
#   export NODE_RANK=0
#   bash scripts/oneflow/launch_pt_text_4node.sh \
#     --web_bundle /path/to/bundle_web_1024 \
#     --code_bundle /path/to/bundle_code_1024 \
#     --output_dir /path/to/ckpts/text_pt_4n \
#     --max_steps 50000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# Ascend env scripts may reference unset vars (e.g. ZSH_VERSION) and will fail under `set -u`.
# Temporarily disable nounset for environment activation.
set +u
source activate_python_env.sh
set -u

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}" || -z "${NODE_RANK:-}" ]]; then
  echo "[ERROR] Please export MASTER_ADDR, MASTER_PORT, NODE_RANK (0..3)."
  exit 2
fi

WEB_BUNDLE=""
CODE_BUNDLE=""
OUTPUT_DIR=""
MAX_STEPS="40000"
MAX_LENGTH="1024"
BS="16"
GA="1"
LR="1e-4"
WARMUP_RATIO="0.01"
NUM_WORKERS="4"
SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="5"
LOGGING_STEPS="20"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/launch_pt_text_4node.sh [options]

Options:
  --web_bundle <path>          Required. Contains: dataset/ + tokenizer/
  --code_bundle <path>         Optional. Contains: dataset/
  --output_dir <path>          Required. Checkpoints output dir
  --max_steps <int>            Default: 40000
  --max_length <int>           Default: 1024
  --per_device_train_batch_size <int>  Default: 16
  --gradient_accumulation_steps <int>  Default: 1
  --learning_rate <float>      Default: 1e-4
  --warmup_ratio <float>       Default: 0.01
  --dataloader_num_workers <int> Default: 4
  --save_steps <int>           Default: 2000
  --save_total_limit <int>     Default: 5
  --logging_steps <int>        Default: 20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --web_bundle) WEB_BUNDLE="$2"; shift 2 ;;
    --code_bundle) CODE_BUNDLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --per_device_train_batch_size) BS="$2"; shift 2 ;;
    --gradient_accumulation_steps) GA="$2"; shift 2 ;;
    --learning_rate) LR="$2"; shift 2 ;;
    --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --dataloader_num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --save_steps) SAVE_STEPS="$2"; shift 2 ;;
    --save_total_limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --logging_steps) LOGGING_STEPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$WEB_BUNDLE" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --web_bundle and --output_dir are required."
  usage
  exit 2
fi

TOKENIZER_DIR="${WEB_BUNDLE%/}/tokenizer"
WEB_DATASET_DIR="${WEB_BUNDLE%/}/dataset"

if [[ -n "$CODE_BUNDLE" ]]; then
  CODE_DATASET_DIR="${CODE_BUNDLE%/}/dataset"
  DATASET_ARGS="${WEB_DATASET_DIR} + ${CODE_DATASET_DIR}"
else
  DATASET_ARGS="${WEB_DATASET_DIR}"
fi

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK"
echo "[INFO] tokenizer: $TOKENIZER_DIR"
echo "[INFO] dataset_args: $DATASET_ARGS"
echo "[INFO] output_dir: $OUTPUT_DIR"

accelerate launch \
  --config_file scripts/accelerate_configs/npu_ddp_4node.yaml \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  examples/oneflow/pt_text.py \
  --output_dir "$OUTPUT_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_DIR" \
  --dataset_args "$DATASET_ARGS" \
  --load_preprocessed_data True \
  --streaming False \
  --max_length "$MAX_LENGTH" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps "$GA" \
  --learning_rate "$LR" \
  --warmup_ratio "$WARMUP_RATIO" \
  --dataloader_num_workers "$NUM_WORKERS" \
  --eval_strategy no --do_eval False \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps "$LOGGING_STEPS" \
  --report_to none \
  --ddp_find_unused_parameters False

