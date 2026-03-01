#!/usr/bin/env python3
"""
Batch image sampling evaluation for OneFlow image-only checkpoints.

Input:
  - a model checkpoint directory
  - a JSONL prompt file (each line: {"id", "category", "prompt"})
  - a VAE decoder (for decoding latents to PNG)

Output:
  - <output_dir>/images/<id>.png   (decoded image per prompt)
  - <output_dir>/report.jsonl      (per-prompt metadata)
  - <output_dir>/report.md         (summary table)

Example:
  python -u scripts/oneflow_image_only/eval_image_sample.py \
    --model_dir data/ckpts/image_only_overfit_1b1/checkpoint-5000 \
    --prompts_file scripts/oneflow/eval_prompts_image_v1.jsonl \
    --vae_id_or_path stabilityai/sd-vae-ft-mse \
    --output_dir outputs/eval_image/image_only_1b1
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import transformers
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

import dllm
from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ScriptArguments:
    model_dir: str = ""
    prompts_file: str = "scripts/oneflow/eval_prompts_image_v1.jsonl"
    output_dir: str = "outputs/oneflow_image_eval"
    seed: int = 42
    device: str = "auto"  # auto|cpu|cuda|npu
    max_prompts: int = 0  # <=0 means all

    # VAE
    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215
    latent_h: int = 0  # inferred if 0
    latent_w: int = 0


@dataclass
class SamplerArgs(OneFlowSamplerConfig):
    dt: float | None = None
    max_steps: int | None = None
    image_num_tokens: int = 256
    temperature: float | None = None
    use_pi_gate: bool | None = None
    max_w: float | None = None
    condition_text_on_time: bool | None = None
    kappa_scheduler_cls: str | None = None
    return_dict: bool = True


def _npu_available() -> bool:
    return bool(hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available())


def _resolve_device(name: str) -> torch.device:
    dev = str(name or "auto").lower()
    if dev == "auto":
        if _npu_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if dev in ("cpu", "cuda", "npu"):
        return torch.device(dev)
    raise ValueError(f"Unknown --device: {name} (expected auto|cpu|cuda|npu)")


def _load_prompts(path: str, max_prompts: int) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if "prompt" not in row:
                raise ValueError(f"Line {i+1} has no 'prompt' field.")
            row.setdefault("id", f"row_{i+1}")
            row.setdefault("category", "general")
            out.append(row)
            if int(max_prompts) > 0 and len(out) >= int(max_prompts):
                break
    return out


def _decode_latents(
    lat_tokens: torch.Tensor,
    vae: AutoencoderKL,
    device: torch.device,
    latent_scale: float,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Decode latent tokens [N, 4] -> image tensor [1, 3, H, W] in [0, 1]."""
    lat_tokens = lat_tokens.to(device=device, dtype=torch.float32)
    n, d = lat_tokens.shape

    if latent_h > 0 and latent_w > 0:
        h, w = int(latent_h), int(latent_w)
        if h * w != n:
            raise ValueError(f"latent_h*latent_w must equal N={n}, got {h}*{w}")
    else:
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(
                f"Cannot infer square latent shape from N={n}. "
                "Please pass --latent_h and --latent_w."
            )
        h, w = side, side

    lat = lat_tokens.reshape(h, w, d).permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
    lat = lat / float(latent_scale)

    with torch.no_grad():
        img = vae.decode(lat).sample  # [-1, 1]
        img = (img / 2 + 0.5).clamp(0, 1)

    return img


def _build_markdown(records: list[dict], model_dir: str, prompts_file: str) -> str:
    total = len(records)
    has_image = sum(1 for r in records if r.get("has_image"))
    lines = []
    lines.append("# OneFlow Image Sampling Eval Report")
    lines.append("")
    lines.append(f"- model_dir: `{model_dir}`")
    lines.append(f"- prompts_file: `{prompts_file}`")
    lines.append(f"- total_prompts: {total}")
    lines.append(f"- images_generated: {has_image}/{total}")
    lines.append("")
    lines.append("| id | category | has_image | image_tokens | output_text_preview |")
    lines.append("|---|---|---:|---:|---|")
    for r in records:
        has_img = "Y" if r.get("has_image") else "N"
        n_tok = r.get("image_token_count", 0)
        preview = str(r.get("output_text", "")).replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        lines.append(f"| {r.get('id')} | {r.get('category')} | {has_img} | {n_tok} | {preview} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerArgs))
    script_args, sampler_args = parser.parse_args_into_dataclasses()
    if not script_args.model_dir:
        raise ValueError("--model_dir is required")

    transformers.set_seed(int(script_args.seed))
    os.makedirs(script_args.output_dir, exist_ok=True)
    img_dir = os.path.join(script_args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    prompts = _load_prompts(script_args.prompts_file, int(script_args.max_prompts))
    if not prompts:
        raise ValueError(f"No prompts found in: {script_args.prompts_file}")

    # Load model
    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_dir)
    model = OneFlowModel.from_pretrained(script_args.model_dir, map_location="cpu").eval()
    device = _resolve_device(script_args.device)
    model = model.to(device)
    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)

    # Resolve sampler config from checkpoint runtime + CLI overrides
    runtime_cfg = load_runtime_config(script_args.model_dir)
    resolved_sampling, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="sampling",
        cli_overrides={
            "scheduler_cls": sampler_args.kappa_scheduler_cls,
            "dt": sampler_args.dt,
            "max_steps": sampler_args.max_steps,
            "temperature": sampler_args.temperature,
            "use_pi_gate": sampler_args.use_pi_gate,
            "append_only": sampler_args.append_only,
            "edit_prompt": sampler_args.edit_prompt,
            "condition_text_on_time": sampler_args.condition_text_on_time,
            "max_w": sampler_args.max_w,
        },
        defaults=DEFAULT_TEXT_SAMPLER_RUNTIME,
    )
    if runtime_cfg is None:
        logger.warning("No oneflow_runtime_config.json found; using built-in sampler defaults.")
    if applied_ckpt_keys:
        logger.info(f"Using checkpoint runtime config for: {sorted(applied_ckpt_keys)}")
    if override_keys:
        logger.warning(f"CLI overrides for: {sorted(override_keys)}")

    sampler_args.kappa_scheduler_cls = str(resolved_sampling["scheduler_cls"])
    sampler_args.dt = float(resolved_sampling["dt"])
    sampler_args.max_steps = int(resolved_sampling["max_steps"])
    sampler_args.temperature = float(resolved_sampling["temperature"])
    sampler_args.use_pi_gate = bool(resolved_sampling["use_pi_gate"])
    sampler_args.append_only = bool(resolved_sampling["append_only"])
    sampler_args.edit_prompt = bool(resolved_sampling["edit_prompt"])
    sampler_args.condition_text_on_time = bool(resolved_sampling["condition_text_on_time"])
    sampler_args.max_w = (
        None if resolved_sampling["max_w"] is None else float(resolved_sampling["max_w"])
    )

    # Load VAE
    logger.info(f"Loading VAE: {script_args.vae_id_or_path}")
    vae = AutoencoderKL.from_pretrained(script_args.vae_id_or_path).to(device).eval()

    # Run evaluation
    out_jsonl = os.path.join(script_args.output_dir, "report.jsonl")
    out_md = os.path.join(script_args.output_dir, "report.md")
    records: list[dict] = []

    image_token = ONEFLOW_IMAGE_TOKEN

    with open(out_jsonl, "w", encoding="utf-8") as fj:
        for i, row in enumerate(prompts):
            prompt_text = str(row["prompt"])
            pid = str(row.get("id", f"row_{i+1}"))
            cat = str(row.get("category", "general"))

            # Append image token if not already present
            if image_token not in prompt_text:
                full_prompt = prompt_text + " " + image_token
            else:
                full_prompt = prompt_text

            prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
            out = sampler.sample([prompt_ids], sampler_args, return_dict=True)
            assert isinstance(out, OneFlowSamplerOutput)

            text = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=False)

            has_image = bool(out.images and len(out.images) > 0)
            image_token_count = 0
            image_path = None

            if has_image:
                lat_tokens = out.images[0]
                image_token_count = int(lat_tokens.shape[0])
                try:
                    img = _decode_latents(
                        lat_tokens, vae, device,
                        script_args.latent_scale,
                        script_args.latent_h,
                        script_args.latent_w,
                    )
                    image_path = os.path.join(img_dir, f"{pid}.png")
                    save_image(img.detach().cpu(), image_path)
                    logger.info(f"[{i+1}/{len(prompts)}] {pid}: decoded image -> {image_path}")
                except Exception as e:
                    logger.warning(f"[{i+1}/{len(prompts)}] {pid}: VAE decode failed: {e}")
                    image_path = None

            rec = {
                "id": pid,
                "category": cat,
                "prompt": prompt_text,
                "full_prompt": full_prompt,
                "output_text": text,
                "has_image": has_image,
                "image_token_count": image_token_count,
                "image_path": image_path,
                "output_num_tokens": int(out.sequences[0].numel()),
            }
            records.append(rec)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fj.flush()

            status = "IMG" if has_image else "no-img"
            logger.info(f"[{i+1}/{len(prompts)}] {pid} ({cat}) [{status}] done.")

    md = _build_markdown(records, script_args.model_dir, script_args.prompts_file)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Wrote JSONL report: {out_jsonl}")
    logger.info(f"Wrote Markdown report: {out_md}")
    logger.info(f"Images saved to: {img_dir}")


if __name__ == "__main__":
    main()
