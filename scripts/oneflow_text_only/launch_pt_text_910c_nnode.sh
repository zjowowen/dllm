#!/usr/bin/env bash
set -euo pipefail

# 910C N-node text-only launcher (no Slurm).
#
# Run the same command on all nodes; only --machine_rank differs.
# Example (N=4):
#   # rank0
#   WANDB_MODE=online bash scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh \
#     --pt_bundle data/offline/pt_text_fineweb_1024_100k \
#     --output_dir data/ckpts/oneflow_text_only_pt_910c_n4_e4000 \
#     --num_machines 4 \
#     --machine_rank 0 \
#     --master_addr 10.119.10.155 \
#     --num_train_epochs 4000 \
#     --save_every_epochs 200
#
#   # rank1 / rank2 / rank3 use same command with --machine_rank 1/2/3.

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
export WANDB_MODE="${WANDB_MODE:-online}"

PT_BUNDLE=""
OUTPUT_DIR=""
INIT_MODEL_DIR=""
RUN_NAME=""
REPORT_TO=""

ACCELERATE_CONFIG="scripts/accelerate_configs/npu_ddp_2node.yaml"
NUM_MACHINES="2"
MACHINE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29530}"
PROCESSES_PER_MACHINE="16"
NUM_PROCESSES=""

MAX_STEPS=""
NUM_TRAIN_EPOCHS="4000"
SAVE_EVERY_EPOCHS="200"
MAX_LENGTH="1024"
BS="24"
GA="1"
LR="1e-4"
WARMUP_RATIO="0.01"
NUM_WORKERS="0"
LOGGING_STEPS="20"
SAVE_STEPS=""
SAVE_TOTAL_LIMIT="30"
LOG_SPLIT_LOSSES="${LOG_SPLIT_LOSSES:-True}"

# ---- training objective defaults (validated via control experiments) ----
TEXT_LOSS_TYPE="ctmc"
CONDITION_TEXT_ON_TIME="True"
NORMALIZE_TEXT_LOSS_BY_LENGTH="True"
MAX_W="20.0"

EXTRA_TRAIN_ARGS=()

usage() {
  cat <<EOF
Usage:
  bash scripts/oneflow_text_only/launch_pt_text_910c_nnode.sh [options] [-- extra_hf_args...]

Required:
  --pt_bundle <path>            Offline PT bundle (contains dataset/ + tokenizer/)
  --output_dir <path>           Checkpoint output directory
  --master_addr <host/ip>       Required when --num_machines > 1

Optional:
  --init_model_dir <path>       Warm-start checkpoint directory
  --run_name <str>              Optional run name
  --report_to <str>             Optional. If omitted: auto (wandb when WANDB_MODE=online, else none)
  --accelerate_config <path>    Default: scripts/accelerate_configs/npu_ddp_2node.yaml
  --num_machines <int>          Default: 2
  --machine_rank <int>          Default: \$NODE_RANK or 0
  --main_process_port <int>     Default: 29530
  --processes_per_machine <int> Default: 16
  --num_processes <int>         Optional override. Default: num_machines * processes_per_machine
  --num_train_epochs <int>      Default: 4000
  --save_every_epochs <int>     Default: 200
  --max_steps <int>             Optional override. If set, takes priority over epoch mode.
  --max_length <int>            Default: 1024
  --per_device_train_batch_size <int> Default: 24
  --gradient_accumulation_steps <int> Default: 1
  --learning_rate <float>       Default: 1e-4
  --warmup_ratio <float>        Default: 0.01
  --dataloader_num_workers <int> Default: 0
  --logging_steps <int>         Default: 20
  --save_steps <int>            Optional override. Default: auto-computed from save_every_epochs.
  --save_total_limit <int>      Default: 30
  --log_split_losses <bool>     Default: True
  --text_loss_type <str>        Default: ctmc (alternatives: paper)
  --condition_text_on_time <bool> Default: True
  --normalize_text_loss_by_length <bool> Default: True
  --max_w <float>               Default: 20.0

Pass-through:
  Place extra HF args after '--', e.g.:
    ... -- --seed 123 --bf16 True
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt_bundle) PT_BUNDLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --init_model_dir) INIT_MODEL_DIR="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --report_to) REPORT_TO="$2"; shift 2 ;;
    --accelerate_config) ACCELERATE_CONFIG="$2"; shift 2 ;;
    --num_machines) NUM_MACHINES="$2"; shift 2 ;;
    --machine_rank|--node_rank) MACHINE_RANK="$2"; shift 2 ;;
    --master_addr) MASTER_ADDR="$2"; shift 2 ;;
    --main_process_port|--master_port) MASTER_PORT="$2"; shift 2 ;;
    --processes_per_machine) PROCESSES_PER_MACHINE="$2"; shift 2 ;;
    --num_processes) NUM_PROCESSES="$2"; shift 2 ;;
    --num_train_epochs) NUM_TRAIN_EPOCHS="$2"; shift 2 ;;
    --save_every_epochs) SAVE_EVERY_EPOCHS="$2"; shift 2 ;;
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
    --text_loss_type) TEXT_LOSS_TYPE="$2"; shift 2 ;;
    --condition_text_on_time) CONDITION_TEXT_ON_TIME="$2"; shift 2 ;;
    --normalize_text_loss_by_length) NORMALIZE_TEXT_LOSS_BY_LENGTH="$2"; shift 2 ;;
    --max_w) MAX_W="$2"; shift 2 ;;
    --) shift; EXTRA_TRAIN_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$PT_BUNDLE" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --pt_bundle and --output_dir are required."
  usage
  exit 2
fi

for v in NUM_MACHINES MACHINE_RANK MASTER_PORT PROCESSES_PER_MACHINE BS GA; do
  if ! [[ "${!v}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --${v,,} must be an integer (got: ${!v})"
    exit 2
  fi
done

if [[ "$NUM_MACHINES" -le 0 ]]; then
  echo "[ERROR] --num_machines must be > 0."
  exit 2
fi
if [[ "$MACHINE_RANK" -lt 0 || "$MACHINE_RANK" -ge "$NUM_MACHINES" ]]; then
  echo "[ERROR] --machine_rank must be in [0, num_machines-1]. Got machine_rank=$MACHINE_RANK num_machines=$NUM_MACHINES"
  exit 2
fi
if [[ "$NUM_MACHINES" -gt 1 && -z "$MASTER_ADDR" ]]; then
  echo "[ERROR] --master_addr is required when --num_machines > 1."
  exit 2
fi

if [[ -z "$NUM_PROCESSES" ]]; then
  NUM_PROCESSES=$((NUM_MACHINES * PROCESSES_PER_MACHINE))
fi
if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [[ "$NUM_PROCESSES" -le 0 ]]; then
  echo "[ERROR] --num_processes must be a positive integer."
  exit 2
fi
if (( NUM_PROCESSES % NUM_MACHINES != 0 )); then
  echo "[ERROR] --num_processes must be divisible by --num_machines. Got num_processes=$NUM_PROCESSES num_machines=$NUM_MACHINES"
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

GLOBAL_BATCH=$((BS * GA * NUM_PROCESSES))
if [[ "$GLOBAL_BATCH" -le 0 ]]; then
  echo "[ERROR] invalid global batch size: $GLOBAL_BATCH"
  exit 2
fi

if [[ -z "$MAX_STEPS" ]]; then
  if ! [[ "$NUM_TRAIN_EPOCHS" =~ ^[0-9]+$ ]] || ! [[ "$SAVE_EVERY_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --num_train_epochs and --save_every_epochs must be positive integers in epoch mode."
    exit 2
  fi
  if [[ "$NUM_TRAIN_EPOCHS" -le 0 || "$SAVE_EVERY_EPOCHS" -le 0 ]]; then
    echo "[ERROR] --num_train_epochs and --save_every_epochs must be > 0."
    exit 2
  fi

  TRAIN_SAMPLES="$(python - "$DATASET_DIR" <<'PY'
import sys
from datasets import load_from_disk

dataset_dir = sys.argv[1]
ds = load_from_disk(dataset_dir)
if "train" not in ds:
    raise SystemExit(f"[ERROR] dataset has no 'train' split: {list(ds.keys())}")
print(len(ds["train"]))
PY
)"

  if ! [[ "$TRAIN_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] failed to resolve train samples from dataset: $TRAIN_SAMPLES"
    exit 2
  fi
  if [[ "$TRAIN_SAMPLES" -le 0 ]]; then
    echo "[ERROR] train split is empty."
    exit 2
  fi

  STEPS_PER_EPOCH=$(((TRAIN_SAMPLES + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
  if [[ "$STEPS_PER_EPOCH" -le 0 ]]; then
    STEPS_PER_EPOCH=1
  fi
  if [[ -z "$SAVE_STEPS" ]]; then
    SAVE_STEPS=$((STEPS_PER_EPOCH * SAVE_EVERY_EPOCHS))
  fi
  if [[ "$SAVE_STEPS" -le 0 ]]; then
    SAVE_STEPS=1
  fi
else
  if [[ -z "$SAVE_STEPS" ]]; then
    SAVE_STEPS="100"
  fi
fi

if [[ -z "$REPORT_TO" ]]; then
  if [[ "${WANDB_MODE:-offline}" == "online" ]]; then
    REPORT_TO="wandb"
  else
    REPORT_TO="none"
  fi
fi

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] tensorboard_dir: $TB_DIR"
echo "[INFO] tokenizer_dir: $TOKENIZER_DIR"
echo "[INFO] dataset_dir: $DATASET_DIR"
echo "[INFO] init_model_dir: ${INIT_MODEL_DIR:-<none>}"
echo "[INFO] distributed: num_machines=$NUM_MACHINES num_processes=$NUM_PROCESSES machine_rank=$MACHINE_RANK"
echo "[INFO] distributed: processes_per_machine=$((NUM_PROCESSES / NUM_MACHINES))"
if [[ "$NUM_MACHINES" -gt 1 || -n "$MASTER_ADDR" ]]; then
  echo "[INFO] rendezvous: master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
fi
echo "[INFO] global_batch_size(per_step): $GLOBAL_BATCH"
if [[ -z "$MAX_STEPS" ]]; then
  echo "[INFO] train_samples: $TRAIN_SAMPLES"
  echo "[INFO] steps_per_epoch(approx): $STEPS_PER_EPOCH"
  echo "[INFO] epoch mode: num_train_epochs=$NUM_TRAIN_EPOCHS save_every_epochs=$SAVE_EVERY_EPOCHS -> save_steps=$SAVE_STEPS"
else
  echo "[INFO] step mode override: max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
fi
echo "[INFO] WANDB_MODE=$WANDB_MODE"
echo "[INFO] report_to=$REPORT_TO"
if [[ "${WANDB_MODE:-offline}" == "online" && "$REPORT_TO" == "none" ]]; then
  echo "[WARN] WANDB_MODE=online but report_to=none, so W&B callback is disabled."
fi

cmd=(
  accelerate launch
  --config_file "$ACCELERATE_CONFIG"
  --num_machines "$NUM_MACHINES"
  --num_processes "$NUM_PROCESSES"
  --machine_rank "$MACHINE_RANK"
)
if [[ "$NUM_MACHINES" -gt 1 || -n "$MASTER_ADDR" ]]; then
  cmd+=(--main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT")
fi

cmd+=(
  examples/oneflow_text_only/pt_text.py
  --output_dir "$OUTPUT_DIR"
  --run_name "${RUN_NAME:-$OUTPUT_DIR}"
  --tokenizer_name_or_path "$TOKENIZER_DIR"
  --dataset_args "$DATASET_DIR"
  --load_preprocessed_data True
  --streaming False
  --max_length "$MAX_LENGTH"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
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
  --text_loss_type "$TEXT_LOSS_TYPE"
  --condition_text_on_time "$CONDITION_TEXT_ON_TIME"
  --normalize_text_loss_by_length "$NORMALIZE_TEXT_LOSS_BY_LENGTH"
  --max_w "$MAX_W"
  --logging_dir "$TB_DIR"
  --report_to "$REPORT_TO"
  --ddp_find_unused_parameters False
)

if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$INIT_MODEL_DIR" ]]; then
  cmd+=(--init_model_dir "$INIT_MODEL_DIR")
fi
if [[ "${#EXTRA_TRAIN_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

