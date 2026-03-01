#!/usr/bin/env bash
set -euo pipefail

# 910C 2-node PT bring-up launcher (no Slurm).
#
# Run the same command on both nodes; only --machine_rank differs.
# Example (rank0):
#   bash scripts/oneflow/launch_pt_text_910c_2node.sh \
#     --pt_bundle data/offline/pt_text_fineweb_1024_100k \
#     --output_dir data/ckpts/oneflow_text_pt_910c_2node_bringup_s200 \
#     --init_model_dir data/ckpts/oneflow_text_pt_910c_continue_from300_s2000/checkpoint-1600 \
#     --master_addr 10.119.10.155 \
#     --machine_rank 0 \
#     --max_steps 200

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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
INIT_MODEL_DIR=""
RUN_NAME=""

ACCELERATE_CONFIG="scripts/accelerate_configs/npu_ddp_2node.yaml"
NUM_MACHINES="2"
NUM_PROCESSES="32"
MACHINE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29530}"

MAX_STEPS="200"
MAX_LENGTH="1024"
BS="8"
GA="1"
LR="1e-4"
WARMUP_RATIO="0.01"
NUM_WORKERS="0"
LOGGING_STEPS="20"
SAVE_STEPS="100"
SAVE_TOTAL_LIMIT="4"
LOG_SPLIT_LOSSES="${LOG_SPLIT_LOSSES:-True}"

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow/launch_pt_text_910c_2node.sh [options]

Required:
  --pt_bundle <path>            Offline PT bundle (contains dataset/ + tokenizer/)
  --output_dir <path>           Checkpoint output directory
  --master_addr <host/ip>       Rank0 host/IP visible by all nodes
  --machine_rank <0|1>          Node rank in this 2-node run

Optional:
  --init_model_dir <path>       Warm-start checkpoint directory
  --run_name <str>              Optional run name
  --accelerate_config <path>    Default: scripts/accelerate_configs/npu_ddp_2node.yaml
  --num_machines <int>          Default: 2
  --num_processes <int>         Default: 32 (2x16 NPUs)
  --main_process_port <int>     Default: 29530
  --max_steps <int>             Default: 200
  --max_length <int>            Default: 1024
  --per_device_train_batch_size <int> Default: 8
  --gradient_accumulation_steps <int> Default: 1
  --learning_rate <float>       Default: 1e-4
  --warmup_ratio <float>        Default: 0.01
  --dataloader_num_workers <int> Default: 0
  --logging_steps <int>         Default: 20
  --save_steps <int>            Default: 100
  --save_total_limit <int>      Default: 4
  --log_split_losses <bool>     Default: True
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt_bundle) PT_BUNDLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --init_model_dir) INIT_MODEL_DIR="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --accelerate_config) ACCELERATE_CONFIG="$2"; shift 2 ;;
    --num_machines) NUM_MACHINES="$2"; shift 2 ;;
    --num_processes) NUM_PROCESSES="$2"; shift 2 ;;
    --machine_rank) MACHINE_RANK="$2"; shift 2 ;;
    --master_addr) MASTER_ADDR="$2"; shift 2 ;;
    --main_process_port) MASTER_PORT="$2"; shift 2 ;;
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
    --log_split_losses) LOG_SPLIT_LOSSES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$PT_BUNDLE" || -z "$OUTPUT_DIR" || -z "$MASTER_ADDR" ]]; then
  echo "[ERROR] --pt_bundle, --output_dir, --master_addr are required."
  usage
  exit 2
fi

if [[ "$MACHINE_RANK" != "0" && "$MACHINE_RANK" != "1" && "$NUM_MACHINES" == "2" ]]; then
  echo "[ERROR] --machine_rank must be 0 or 1 when --num_machines=2 (got: $MACHINE_RANK)"
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
if [[ -n "$INIT_MODEL_DIR" && ! -d "$INIT_MODEL_DIR" ]]; then
  echo "[ERROR] Missing init model dir: $INIT_MODEL_DIR"
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$TB_DIR"

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] tensorboard_dir: $TB_DIR"
echo "[INFO] tokenizer_dir: $TOKENIZER_DIR"
echo "[INFO] dataset_dir: $DATASET_DIR"
echo "[INFO] init_model_dir: ${INIT_MODEL_DIR:-<none>}"
echo "[INFO] distributed: num_machines=$NUM_MACHINES num_processes=$NUM_PROCESSES machine_rank=$MACHINE_RANK"
echo "[INFO] rendezvous: master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
echo "[INFO] WANDB_MODE=$WANDB_MODE (offline expected)"

cmd=(accelerate launch
  --config_file "$ACCELERATE_CONFIG"
  --num_machines "$NUM_MACHINES"
  --num_processes "$NUM_PROCESSES"
  --machine_rank "$MACHINE_RANK"
  --main_process_ip "$MASTER_ADDR"
  --main_process_port "$MASTER_PORT"
  examples/oneflow/pt_text.py
  --output_dir "$OUTPUT_DIR"
  --run_name "${RUN_NAME:-$OUTPUT_DIR}"
  --tokenizer_name_or_path "$TOKENIZER_DIR"
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
  --report_to none
  --ddp_find_unused_parameters False
)

if [[ -n "$INIT_MODEL_DIR" ]]; then
  cmd+=(--init_model_dir "$INIT_MODEL_DIR")
fi

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

