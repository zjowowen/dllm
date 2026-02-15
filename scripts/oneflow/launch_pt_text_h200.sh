#!/usr/bin/env bash
set -euo pipefail

# One-click PT launch wrapper for offline H200 (CUDA).
#
# Example:
#   bash scripts/oneflow/launch_pt_text_h200.sh \
#     --pt_bundle /path/to/pt_text_fineweb_edu_100k \
#     --output_dir data/ckpts/oneflow_text_pt_h200_fineweb \
#     --max_steps 2000 \
#     --num_processes 8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Optional environment activation:
# - Prefer init_env.sh for H200 (per project note)
# - Fallback to activate_python_env.sh for compatibility
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

PT_BUNDLE=""
OUTPUT_DIR=""
ACCELERATE_CONFIG="scripts/accelerate_configs/ddp.yaml"
NUM_PROCESSES="8"
MAIN_PROCESS_PORT="29500"

MAX_STEPS="2000"
MAX_LENGTH="1024"
BS="8"
GA="1"
LR="1e-4"
WARMUP_RATIO="0.01"
NUM_WORKERS="4"
LOGGING_STEPS="20"
SAVE_STEPS="500"
SAVE_TOTAL_LIMIT="5"
RUN_NAME=""
LOG_SPLIT_LOSSES="${LOG_SPLIT_LOSSES:-True}"
MODEL_DIM="512"
MODEL_DEPTH="8"
MODEL_HEADS="8"
MODEL_DIM_HEAD="64"
MODEL_DIM_LATENT="4"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/launch_pt_text_h200.sh [options]

Required:
  --pt_bundle <path>          Offline PT bundle directory (contains dataset/ + tokenizer/)
  --output_dir <path>         Checkpoint output directory

Optional:
  --accelerate_config <path>  Default: scripts/accelerate_configs/ddp.yaml
  --num_processes <int>       Default: 8
  --main_process_port <int>   Default: 29500
  --max_steps <int>           Default: 2000
  --max_length <int>          Default: 1024
  --per_device_train_batch_size <int> Default: 8
  --gradient_accumulation_steps <int> Default: 1
  --learning_rate <float>     Default: 1e-4
  --warmup_ratio <float>      Default: 0.01
  --dataloader_num_workers <int> Default: 4
  --logging_steps <int>       Default: 20
  --save_steps <int>          Default: 500
  --save_total_limit <int>    Default: 5
  --run_name <str>            Optional run name
  --log_split_losses <bool>   Default: True (log loss_text_pi/lam/tok)
  --dim <int>                 Default: 512
  --depth <int>               Default: 8
  --heads <int>               Default: 8
  --dim_head <int>            Default: 64
  --dim_latent <int>          Default: 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt_bundle) PT_BUNDLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
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
    --log_split_losses) LOG_SPLIT_LOSSES="$2"; shift 2 ;;
    --dim) MODEL_DIM="$2"; shift 2 ;;
    --depth) MODEL_DEPTH="$2"; shift 2 ;;
    --heads) MODEL_HEADS="$2"; shift 2 ;;
    --dim_head) MODEL_DIM_HEAD="$2"; shift 2 ;;
    --dim_latent) MODEL_DIM_LATENT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$PT_BUNDLE" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --pt_bundle and --output_dir are required."
  usage
  exit 2
fi

TOKENIZER_DIR="${PT_BUNDLE%/}/tokenizer"
DATASET_DIR="${PT_BUNDLE%/}/dataset"
TB_DIR="${OUTPUT_DIR%/}/tensorboard"

if [[ ! -d "$TOKENIZER_DIR" ]]; then
  echo "[ERROR] Missing tokenizer dir: $TOKENIZER_DIR"
  exit 2
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[ERROR] Missing dataset dir: $DATASET_DIR"
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$TB_DIR"

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] tensorboard_dir: $TB_DIR"
echo "[INFO] tokenizer_dir: $TOKENIZER_DIR"
echo "[INFO] dataset_dir: $DATASET_DIR"
echo "[INFO] WANDB_MODE=$WANDB_MODE (offline expected)"
echo "[INFO] log_split_losses: $LOG_SPLIT_LOSSES"
echo "[INFO] model: dim=${MODEL_DIM}, depth=${MODEL_DEPTH}, heads=${MODEL_HEADS}, dim_head=${MODEL_DIM_HEAD}, dim_latent=${MODEL_DIM_LATENT}"
echo "[INFO] Use TensorBoard: tensorboard --logdir $TB_DIR"

cmd=(accelerate launch
  --config_file "$ACCELERATE_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --main_process_port "$MAIN_PROCESS_PORT"
  examples/oneflow/pt_text.py
  --output_dir "$OUTPUT_DIR"
  --run_name "${RUN_NAME:-$OUTPUT_DIR}"
  --tokenizer_name_or_path "$TOKENIZER_DIR"
  --dim "$MODEL_DIM"
  --depth "$MODEL_DEPTH"
  --heads "$MODEL_HEADS"
  --dim_head "$MODEL_DIM_HEAD"
  --dim_latent "$MODEL_DIM_LATENT"
  --dataset_args "$DATASET_DIR"
  --load_preprocessed_data True
  --streaming False
  --max_length "$MAX_LENGTH"
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
  --log_split_losses "$LOG_SPLIT_LOSSES"
  --logging_dir "$TB_DIR"
  --report_to wandb tensorboard
  --ddp_find_unused_parameters False
)

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
