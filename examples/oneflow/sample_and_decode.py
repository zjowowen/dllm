"""
Sample from a OneFlow checkpoint and decode image latents to PNGs using a VAE.

This is useful on an offline cluster if you copy the VAE weights directory
ahead of time (e.g., download `stabilityai/sd-vae-ft-mse` on an online machine
and `save_pretrained` to a local path).

Example:
  python -u examples/oneflow/sample_and_decode.py \
    --model_dir "/path/to/checkpoint-final" \
    --prompt "a photo of flower 10 <|oneflow_image|>" \
    --vae_id_or_path "/path/to/sd-vae-ft-mse" \
    --output_dir "/path/to/vis" \
    --latent_h 16 --latent_w 16
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import transformers
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput


@dataclass
class Args:
    model_dir: str = None  # overwrite this
    prompt: str = "a photo of a flower <|oneflow_image|>"
    output_dir: str = "outputs/oneflow_decode"

    # VAE
    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215

    # latent token layout (N = H*W)
    latent_h: int = 0
    latent_w: int = 0

    # sampling
    dt: float = 0.1
    max_steps: int = 50
    temperature: float = 0.0
    use_pi_gate: bool = True
    edit_prompt: bool = False
    image_num_tokens: int = 256

    seed: int = 42


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")

    transformers.set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    model = OneFlowModel.from_pretrained(args.model_dir, map_location="cpu").eval().to(device)

    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)
    cfg = OneFlowSamplerConfig(
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        temperature=float(args.temperature),
        use_pi_gate=bool(args.use_pi_gate),
        edit_prompt=bool(args.edit_prompt),
        image_num_tokens=int(args.image_num_tokens),
        return_dict=True,
    )

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    out = sampler.sample([prompt_ids], cfg, return_dict=True)
    assert isinstance(out, OneFlowSamplerOutput)

    text = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=False)
    text_path = os.path.join(args.output_dir, "text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n=== Text ===")
    print(text)
    print("\nSaved:", text_path)

    if not out.images:
        print("\nNo image latents were produced. Tip: include `<|oneflow_image|>` in the prompt.")
        return

    print("\nLoading VAE:", args.vae_id_or_path)
    vae = AutoencoderKL.from_pretrained(args.vae_id_or_path).to(device).eval()

    for i, lat_tokens in enumerate(out.images):
        lat_tokens = lat_tokens.to(device=device, dtype=torch.float32)  # [N,4]
        n, d = lat_tokens.shape

        if args.latent_h > 0 and args.latent_w > 0:
            h, w = int(args.latent_h), int(args.latent_w)
            if h * w != n:
                raise ValueError(f"--latent_h*--latent_w must equal N={n}, got {h}*{w}")
        else:
            side = int(math.isqrt(n))
            if side * side != n:
                raise ValueError(
                    f"Cannot infer square latent shape from N={n}. "
                    "Please pass --latent_h and --latent_w."
                )
            h, w = side, side

        # tokens (H*W,4) -> (1,4,H,W)
        lat = lat_tokens.reshape(h, w, d).permute(2, 0, 1).unsqueeze(0)
        lat = lat / float(args.latent_scale)

        with torch.no_grad():
            img = vae.decode(lat).sample  # [-1,1]
            img = (img / 2 + 0.5).clamp(0, 1)

        out_path = os.path.join(args.output_dir, f"image_{i}.png")
        save_image(img.detach().cpu(), out_path)
        print(f"[image {i}] tokens={tuple(lat_tokens.shape)} -> decoded={tuple(img.shape)} -> {out_path}")


if __name__ == "__main__":
    main()


