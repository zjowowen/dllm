#!/usr/bin/env python3
"""
Interleaved sampling script for OneFlow checkpoints.

Supports three modes:
  - text_only:         Generate text from a text prompt (no image generation).
  - image_conditioned: Generate an image conditioned on a text prompt.
  - interleaved:       Generate both text and image tokens in an interleaved fashion.

Example:
  # Text-only
  python -u examples/oneflow_interleaved/sample_interleaved.py \
    --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
    --mode text_only \
    --prompt "OneFlow is a generative model" \
    --output_dir outputs/interleaved_text

  # Image-conditioned
  python -u examples/oneflow_interleaved/sample_interleaved.py \
    --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
    --mode image_conditioned \
    --prompt "a photo of a flower" \
    --vae_id_or_path stabilityai/sd-vae-ft-mse \
    --output_dir outputs/interleaved_image

  # Full interleaved
  python -u examples/oneflow_interleaved/sample_interleaved.py \
    --model_dir data/ckpts/interleaved_2b1_baseline/checkpoint-5000 \
    --mode interleaved \
    --prompt "a photo of a flower" \
    --vae_id_or_path stabilityai/sd-vae-ft-mse \
    --output_dir outputs/interleaved_full
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import transformers
from torchvision.utils import save_image

from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN


@dataclass
class Args:
    model_dir: str = ""
    prompt: str = "a photo of a flower"
    mode: str = "interleaved"  # text_only | image_conditioned | interleaved
    output_dir: str = "outputs/oneflow_interleaved_sample"

    # VAE (only needed for image_conditioned and interleaved modes)
    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215
    latent_h: int = 0  # inferred if 0
    latent_w: int = 0

    # Sampling
    dt: float = 0.05
    max_steps: int = 40
    temperature: float = 0.0
    use_pi_gate: bool = True
    edit_prompt: bool = False
    append_only: bool = False
    condition_text_on_time: bool = True
    max_w: float = 20.0
    image_num_tokens: int = 256
    max_new_tokens: int = 256
    max_seq_len: int = 512
    kappa_scheduler_cls: str = ""

    seed: int = 42
    device: str = "auto"  # auto|cpu|cuda|npu


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
    raise ValueError(f"Unknown --device: {name}")


def _decode_latents(
    lat_tokens: torch.Tensor,
    vae,
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


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")

    valid_modes = ("text_only", "image_conditioned", "interleaved")
    if args.mode not in valid_modes:
        raise ValueError(f"--mode must be one of {valid_modes}, got: {args.mode}")

    transformers.set_seed(int(args.seed))
    device = _resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_dir}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    model = OneFlowModel.from_pretrained(args.model_dir, map_location="cpu").eval().to(device)
    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)

    # Build sampler config with runtime resolution
    runtime_cfg = load_runtime_config(args.model_dir)
    resolved_sampling, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="sampling",
        cli_overrides={
            "scheduler_cls": args.kappa_scheduler_cls or None,
            "dt": args.dt,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "use_pi_gate": args.use_pi_gate,
            "append_only": args.append_only,
            "edit_prompt": args.edit_prompt,
            "condition_text_on_time": args.condition_text_on_time,
            "max_w": args.max_w,
        },
        defaults=DEFAULT_TEXT_SAMPLER_RUNTIME,
    )
    if runtime_cfg is None:
        print("[WARN] No oneflow_runtime_config.json found; using built-in defaults.")
    if applied_ckpt_keys:
        print(f"[INFO] Using checkpoint runtime config for: {sorted(applied_ckpt_keys)}")
    if override_keys:
        print(f"[INFO] CLI overrides for: {sorted(override_keys)}")

    cfg = OneFlowSamplerConfig(
        dt=float(resolved_sampling["dt"]),
        max_steps=int(resolved_sampling["max_steps"]),
        temperature=float(resolved_sampling["temperature"]),
        use_pi_gate=bool(resolved_sampling["use_pi_gate"]),
        append_only=bool(resolved_sampling["append_only"]),
        edit_prompt=bool(resolved_sampling["edit_prompt"]),
        condition_text_on_time=bool(resolved_sampling["condition_text_on_time"]),
        max_w=None if resolved_sampling["max_w"] is None else float(resolved_sampling["max_w"]),
        kappa_scheduler_cls=str(resolved_sampling["scheduler_cls"]),
        image_num_tokens=int(args.image_num_tokens),
        max_new_tokens=int(args.max_new_tokens),
        max_seq_len=int(args.max_seq_len),
        return_dict=True,
    )

    # Prepare prompt based on mode
    prompt_text = args.prompt
    image_token = ONEFLOW_IMAGE_TOKEN

    if args.mode == "text_only":
        # No image token; pure text generation
        full_prompt = prompt_text
        cfg.image_num_tokens = 0
    elif args.mode == "image_conditioned":
        # Append image token for image generation
        if image_token not in prompt_text:
            full_prompt = prompt_text + " " + image_token
        else:
            full_prompt = prompt_text
    elif args.mode == "interleaved":
        # Append image token for interleaved generation
        if image_token not in prompt_text:
            full_prompt = prompt_text + " " + image_token
        else:
            full_prompt = prompt_text
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Prompt: {prompt_text}")
    print(f"Full prompt (to tokenizer): {full_prompt}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Sample
    prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    out = sampler.sample([prompt_ids], cfg, return_dict=True)
    assert isinstance(out, OneFlowSamplerOutput)

    # Save text output
    text = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=False)
    text_clean = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=True)
    text_path = os.path.join(args.output_dir, "text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Prompt: {prompt_text}\n")
        f.write(f"Full prompt: {full_prompt}\n")
        f.write(f"\n=== Raw output (with special tokens) ===\n{text}\n")
        f.write(f"\n=== Clean output ===\n{text_clean}\n")

    print("=== Text Output ===")
    print(text_clean)
    print(f"\nSaved text: {text_path}")

    # Decode images if present
    has_images = bool(out.images and len(out.images) > 0)
    if has_images:
        needs_vae = args.mode in ("image_conditioned", "interleaved")
        if needs_vae:
            print(f"\nLoading VAE: {args.vae_id_or_path}")
            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(args.vae_id_or_path).to(device).eval()

            for i, lat_tokens in enumerate(out.images):
                try:
                    img = _decode_latents(
                        lat_tokens, vae, device,
                        args.latent_scale, args.latent_h, args.latent_w,
                    )
                    img_path = os.path.join(args.output_dir, f"image_{i}.png")
                    save_image(img.detach().cpu(), img_path)
                    print(f"[image {i}] tokens={tuple(lat_tokens.shape)} -> {img_path}")
                except Exception as e:
                    print(f"[image {i}] VAE decode failed: {e}")
        else:
            print(f"\n[INFO] {len(out.images)} image latent block(s) produced but --mode=text_only; skipping decode.")
            for i, lat_tokens in enumerate(out.images):
                print(f"  [image {i}] shape={tuple(lat_tokens.shape)}")
    else:
        if args.mode != "text_only":
            print("\n[WARN] No image latents produced. Check prompt and model checkpoint.")

    # Save metadata
    meta = {
        "mode": args.mode,
        "prompt": prompt_text,
        "full_prompt": full_prompt,
        "model_dir": args.model_dir,
        "output_num_tokens": int(out.sequences[0].numel()),
        "num_images": len(out.images) if out.images else 0,
        "image_token_counts": [int(img.shape[0]) for img in out.images] if out.images else [],
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metadata: {meta_path}")


if __name__ == "__main__":
    main()
