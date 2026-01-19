#!/usr/bin/env bash
set -euo pipefail

# Batch overfit evaluation across multiple checkpoints for a fixed training sample.
#
# This is a convenience wrapper around:
#   examples/oneflow/overfit_eval_wds.py
#
# Example:
#   bash scripts/oneflow/batch_overfit_eval_wds.sh \
#     --ckpt_root data/ckpts/stage3b_mm_latents_128_flower32_ft \
#     --wds_shards data/latents_128_bundle/wds_latents_flower32 \
#     --sample_key 000070352 \
#     --out_base data/vis/overfit_trend_000070352 \
#     --device cpu

CKPT_ROOT=""
WDS_SHARDS=""
SAMPLE_KEY=""
OUT_BASE=""
VAE="stabilityai/sd-vae-ft-mse"
LATENT_H=16
LATENT_W=16
IMAGE_NUM_TOKENS=256
DT=0.02
MAX_STEPS=200
DEVICE="cpu"
VAE_DEVICE=""

CHECKPOINTS=( "checkpoint-2000" "checkpoint-4000" "checkpoint-6000" "checkpoint-8000" "checkpoint-10000" "checkpoint-12000" "checkpoint-14000" "checkpoint-16000" "checkpoint-18000" "checkpoint-20000" "checkpoint-final")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_root) CKPT_ROOT="$2"; shift 2;;
    --wds_shards) WDS_SHARDS="$2"; shift 2;;
    --sample_key) SAMPLE_KEY="$2"; shift 2;;
    --out_base) OUT_BASE="$2"; shift 2;;
    --vae_id_or_path) VAE="$2"; shift 2;;
    --latent_h) LATENT_H="$2"; shift 2;;
    --latent_w) LATENT_W="$2"; shift 2;;
    --image_num_tokens) IMAGE_NUM_TOKENS="$2"; shift 2;;
    --dt) DT="$2"; shift 2;;
    --max_steps) MAX_STEPS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --vae_device) VAE_DEVICE="$2"; shift 2;;
    --checkpoints)
      shift
      CHECKPOINTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CHECKPOINTS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CKPT_ROOT}" || -z "${WDS_SHARDS}" || -z "${SAMPLE_KEY}" || -z "${OUT_BASE}" ]]; then
  echo "Usage: $0 --ckpt_root <dir> --wds_shards <dir/glob> --sample_key <key> --out_base <dir> [--device cpu|npu] [--vae_device cpu]" >&2
  exit 2
fi

mkdir -p "${OUT_BASE}"

for ckpt in "${CHECKPOINTS[@]}"; do
  echo ""
  echo "==== ${ckpt} ===="
  out_dir="${OUT_BASE}/${ckpt}"
  rm -rf "${out_dir}"
  if [[ -n "${VAE_DEVICE}" ]]; then
    python -u examples/oneflow/overfit_eval_wds.py \
      --model_dir "${CKPT_ROOT}/${ckpt}" \
      --wds_shards "${WDS_SHARDS}" \
      --sample_key "${SAMPLE_KEY}" \
      --vae_id_or_path "${VAE}" \
      --output_dir "${out_dir}" \
      --latent_h "${LATENT_H}" --latent_w "${LATENT_W}" \
      --image_num_tokens "${IMAGE_NUM_TOKENS}" \
      --dt "${DT}" --max_steps "${MAX_STEPS}" \
      --device "${DEVICE}" \
      --vae_device "${VAE_DEVICE}"
  else
    python -u examples/oneflow/overfit_eval_wds.py \
      --model_dir "${CKPT_ROOT}/${ckpt}" \
      --wds_shards "${WDS_SHARDS}" \
      --sample_key "${SAMPLE_KEY}" \
      --vae_id_or_path "${VAE}" \
      --output_dir "${out_dir}" \
      --latent_h "${LATENT_H}" --latent_w "${LATENT_W}" \
      --image_num_tokens "${IMAGE_NUM_TOKENS}" \
      --dt "${DT}" --max_steps "${MAX_STEPS}" \
      --device "${DEVICE}"
  fi
done

python - "${SAMPLE_KEY}" "${OUT_BASE}" "${CHECKPOINTS[@]}" <<'PY'
import json, os, sys

sample_key = sys.argv[1]
base = sys.argv[2]
ckpts = sys.argv[3:]

print("\nsample_key", sample_key)
print("output_base", base)
print("\nckpt\tlatent_mse\tpixel_psnr\tpixel_mse")
for ck in ckpts:
    p = os.path.join(base, ck, f"metrics_{sample_key}.json")
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    print(f"{ck}\t{m['latent_mse']:.6f}\t{m['pixel_psnr']:.4f}\t{m['pixel_mse']:.6f}")
PY


