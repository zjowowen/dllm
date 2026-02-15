#!/usr/bin/env bash
set -euo pipefail

# One-click pseudo-SFT launch wrapper for offline H200 (CUDA).
#
# Example:
#   bash scripts/oneflow/launch_sft_text_h200.sh \
#     --init_model_dir data/ckpts/oneflow_text_pt_h200_fineweb/checkpoint-final \
#     --sft_bundle data/offline/sft_text_pseudo_from_pt_fineweb_100k \
#     --output_dir data/ckpts/oneflow_text_sft_h200_pseudo \
#     --max_steps 1000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Optional environment activation (same behavior as PT wrapper).
if [[ "${ONEFLOW_SOURCE_INIT_ENV:-1}" == "1" ]]; then
  if [[ -f "${ROOT_DIR}/init_env.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/init_env.sh"
    set -u
  elif [[ -f "${ROOT_DIR}/activate_python_env.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/activate_python_env.sh"
    set -u
  fi
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

INIT_MODEL_DIR=""
SFT_BUNDLE=""
OUTPUT_DIR=""
TOKENIZER_DIR=""

ACCELERATE_CONFIG="scripts/accelerate_configs/ddp.yaml"
NUM_PROCESSES="8"
MAIN_PROCESS_PORT="29500"

MAX_STEPS="1000"
MAX_LENGTH="1024"
BS="4"
GA="1"
LR="2e-5"
WARMUP_RATIO="0.03"
NUM_WORKERS="4"
LOGGING_STEPS="20"
SAVE_STEPS="200"
SAVE_TOTAL_LIMIT="5"
RUN_NAME=""
MODEL_DIM="512"
MODEL_DEPTH="8"
MODEL_HEADS="8"
MODEL_DIM_HEAD="64"
MODEL_DIM_LATENT="4"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/launch_sft_text_h200.sh [options]

Required:
  --init_model_dir <path>      PT checkpoint directory (e.g. checkpoint-final)
  --sft_bundle <path>          Pseudo-SFT bundle (contains dataset/ + tokenizer/)
  --output_dir <path>          SFT output directory

Optional:
  --tokenizer_dir <path>       Override tokenizer path; default: <sft_bundle>/tokenizer
  --accelerate_config <path>   Default: scripts/accelerate_configs/ddp.yaml
  --num_processes <int>        Default: 8
  --main_process_port <int>    Default: 29500
  --max_steps <int>            Default: 1000
  --max_length <int>           Default: 1024
  --per_device_train_batch_size <int> Default: 4
  --gradient_accumulation_steps <int> Default: 1
  --learning_rate <float>      Default: 2e-5
  --warmup_ratio <float>       Default: 0.03
  --dataloader_num_workers <int> Default: 4
  --logging_steps <int>        Default: 20
  --save_steps <int>           Default: 200
  --save_total_limit <int>     Default: 5
  --run_name <str>             Optional run name
  --dim <int>                  Default: 512
  --depth <int>                Default: 8
  --heads <int>                Default: 8
  --dim_head <int>             Default: 64
  --dim_latent <int>           Default: 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init_model_dir) INIT_MODEL_DIR="$2"; shift 2 ;;
    --sft_bundle) SFT_BUNDLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tokenizer_dir) TOKENIZER_DIR="$2"; shift 2 ;;
    --accelerate_config) ACCELERATE_CONFIG="$2"; shift 2 ;;
    --num_processes) NUM_PROCESSES="$2"; shift 2 ;;
    --main_process_port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --per_device_train_batch_size) BS="$2"; shift 2 ;;
    --gradient_accumulation_steps) GA="$2"; shift 2 ;;
    --learning_rate) LR="$2"; shift 2 ;;
    --warmup_ratio) WARMUP_RATIO="$2"; shift 2 ;;
    --dataloader_num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --logging_steps) LOGGING_STEPS="$2"; shift 2 ;;
    --save_steps) SAVE_STEPS="$2"; shift 2 ;;
    --save_total_limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --dim) MODEL_DIM="$2"; shift 2 ;;
    --depth) MODEL_DEPTH="$2"; shift 2 ;;
    --heads) MODEL_HEADS="$2"; shift 2 ;;
    --dim_head) MODEL_DIM_HEAD="$2"; shift 2 ;;
    --dim_latent) MODEL_DIM_LATENT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$INIT_MODEL_DIR" || -z "$SFT_BUNDLE" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --init_model_dir, --sft_bundle, --output_dir are required."
  usage
  exit 2
fi

SFT_DATASET_DIR="${SFT_BUNDLE%/}/dataset"
if [[ -z "$TOKENIZER_DIR" ]]; then
  TOKENIZER_DIR="${SFT_BUNDLE%/}/tokenizer"
fi
TB_DIR="${OUTPUT_DIR%/}/tensorboard"

if [[ ! -d "$INIT_MODEL_DIR" ]]; then
  echo "[ERROR] Missing init model dir: $INIT_MODEL_DIR"
  exit 2
fi
if [[ ! -d "$SFT_DATASET_DIR" ]]; then
  echo "[ERROR] Missing SFT dataset dir: $SFT_DATASET_DIR"
  exit 2
fi
if [[ ! -d "$TOKENIZER_DIR" ]]; then
  echo "[ERROR] Missing tokenizer dir: $TOKENIZER_DIR"
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$TB_DIR"

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] tensorboard_dir: $TB_DIR"
echo "[INFO] init_model_dir: $INIT_MODEL_DIR"
echo "[INFO] sft_dataset_dir: $SFT_DATASET_DIR"
echo "[INFO] tokenizer_dir: $TOKENIZER_DIR"
echo "[INFO] WANDB_MODE=$WANDB_MODE (offline expected)"
echo "[INFO] model args: dim=${MODEL_DIM}, depth=${MODEL_DEPTH}, heads=${MODEL_HEADS}, dim_head=${MODEL_DIM_HEAD}, dim_latent=${MODEL_DIM_LATENT}"
echo "[INFO] Use TensorBoard: tensorboard --logdir $TB_DIR"

cmd=(accelerate launch
  --config_file "$ACCELERATE_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PROCESS_PORT"
  examples/oneflow/sft_mm.py
  --output_dir "$OUTPUT_DIR"
  --run_name "${RUN_NAME:-$OUTPUT_DIR}"
  --init_model_dir "$INIT_MODEL_DIR"
  --dim "$MODEL_DIM"
  --depth "$MODEL_DEPTH"
  --heads "$MODEL_HEADS"
  --dim_head "$MODEL_DIM_HEAD"
  --dim_latent "$MODEL_DIM_LATENT"
  --tokenizer_name_or_path "$TOKENIZER_DIR"
  --dataset_args "$SFT_DATASET_DIR"
  --load_preprocessed_data True
  --mask_prompt_loss True
  --max_length "$MAX_LENGTH"
  --truncation right
  --max_steps "$MAX_STEPS"
  --per_device_train_batch_size "$BS"
  --gradient_accumulation_steps "$GA"
  --learning_rate "$LR"
  --warmup_ratio "$WARMUP_RATIO"
  --dataloader_num_workers "$NUM_WORKERS"
  --eval_strategy no
  --do_eval False
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --logging_steps "$LOGGING_STEPS"
  --logging_dir "$TB_DIR"
  --report_to wandb tensorboard
  --ddp_find_unused_parameters False
)

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
