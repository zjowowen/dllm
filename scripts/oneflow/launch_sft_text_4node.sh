#!/usr/bin/env bash
set -euo pipefail

# One-click 4-node (64 NPU) SFT launch wrapper (text-only).
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
#   bash scripts/oneflow/launch_sft_text_4node.sh \
#     --init_model_dir /path/to/pt/checkpoint-final \
#     --tokenizer_dir /path/to/bundle_web_1024/tokenizer \
#     --sft_bundle /path/to/bundle_sft_ultrachat_1024 \
#     --sft_bundle2 /path/to/bundle_sft_opc_python_1024 \
#     --output_dir /path/to/ckpts/text_sft_4n \
#     --max_steps 5000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source activate_python_env.sh

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}" || -z "${NODE_RANK:-}" ]]; then
  echo "[ERROR] Please export MASTER_ADDR, MASTER_PORT, NODE_RANK (0..3)."
  exit 2
fi

INIT_MODEL_DIR=""
TOKENIZER_DIR=""
SFT_BUNDLE=""
SFT_BUNDLE2=""
OUTPUT_DIR=""

MAX_STEPS="5000"
MAX_LENGTH="1024"
BS="4"
GA="1"
LR="2e-5"
WARMUP_RATIO="0.03"
NUM_WORKERS="4"
SAVE_STEPS="500"
SAVE_TOTAL_LIMIT="5"
LOGGING_STEPS="20"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/launch_sft_text_4node.sh [options]

Options:
  --init_model_dir <path>      Required. PT checkpoint (e.g. checkpoint-final)
  --tokenizer_dir <path>       Required. Tokenizer dir (usually PT bundle tokenizer/)
  --sft_bundle <path>          Required. Contains dataset/ (prepared by prepare_sft_bundle.sh)
  --sft_bundle2 <path>         Optional. Another SFT bundle (dataset/)
  --output_dir <path>          Required. Checkpoints output dir

  --max_steps <int>            Default: 5000
  --max_length <int>           Default: 1024
  --per_device_train_batch_size <int>  Default: 4
  --gradient_accumulation_steps <int>  Default: 1
  --learning_rate <float>      Default: 2e-5
  --warmup_ratio <float>       Default: 0.03
  --dataloader_num_workers <int> Default: 4
  --save_steps <int>           Default: 500
  --save_total_limit <int>     Default: 5
  --logging_steps <int>        Default: 20
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init_model_dir) INIT_MODEL_DIR="$2"; shift 2 ;;
    --tokenizer_dir) TOKENIZER_DIR="$2"; shift 2 ;;
    --sft_bundle) SFT_BUNDLE="$2"; shift 2 ;;
    --sft_bundle2) SFT_BUNDLE2="$2"; shift 2 ;;
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

if [[ -z "$INIT_MODEL_DIR" || -z "$TOKENIZER_DIR" || -z "$SFT_BUNDLE" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --init_model_dir, --tokenizer_dir, --sft_bundle, --output_dir are required."
  usage
  exit 2
fi

SFT_DATASET_DIR="${SFT_BUNDLE%/}/dataset"
if [[ -n "$SFT_BUNDLE2" ]]; then
  SFT_DATASET_DIR2="${SFT_BUNDLE2%/}/dataset"
  DATASET_ARGS="${SFT_DATASET_DIR} + ${SFT_DATASET_DIR2}"
else
  DATASET_ARGS="${SFT_DATASET_DIR}"
fi

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK"
echo "[INFO] init_model_dir: $INIT_MODEL_DIR"
echo "[INFO] tokenizer_dir: $TOKENIZER_DIR"
echo "[INFO] dataset_args: $DATASET_ARGS"
echo "[INFO] output_dir: $OUTPUT_DIR"

accelerate launch \
  --config_file scripts/accelerate_configs/npu_ddp_4node.yaml \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  examples/oneflow/sft_mm.py \
  --output_dir "$OUTPUT_DIR" \
  --init_model_dir "$INIT_MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_DIR" \
  --dataset_args "$DATASET_ARGS" \
  --load_preprocessed_data True \
  --mask_prompt_loss True \
  --max_length "$MAX_LENGTH" \
  --truncation right \
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

