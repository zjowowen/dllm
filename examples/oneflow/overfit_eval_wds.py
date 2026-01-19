#!/usr/bin/env python3
"""
Overfit evaluation tool for OneFlow latents training.

It compares, for a *training sample*:
  - GT image decoded from dataset latent (npy)
  - Generated image decoded from sampler output

This helps answer:
  - Is the dataset itself decodable and meaningful?
  - Does GT-vs-Gen improve as training proceeds (overfitting trend)?
  - Is train/eval tokenization / latent conventions mismatched?

Example:
  python -u examples/oneflow/overfit_eval_wds.py \
    --model_dir data/ckpts/stage3b_mm_latents_128_flower32_ft/checkpoint-400 \
    --wds_shards data/latents_128_bundle/wds_latents_flower32 \
    --sample_index 0 \
    --vae_id_or_path stabilityai/sd-vae-ft-mse \
    --output_dir data/vis/overfit_eval_ckpt400 \
    --latent_h 16 --latent_w 16 --image_num_tokens 256
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import transformers
import webdataset as wds
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_TOKEN
from dllm.pipelines.oneflow.models import OneFlowModel


def expand_shards(spec: str) -> list[str]:
    spec = str(spec)
    if os.path.isdir(spec):
        out = [os.path.join(spec, n) for n in sorted(os.listdir(spec)) if n.endswith(".tar")]
        if not out:
            raise FileNotFoundError(f"No .tar shards found in dir: {spec}")
        return out
    if os.path.isfile(spec) and spec.endswith(".txt"):
        out: list[str] = []
        with open(spec, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
        if not out:
            raise FileNotFoundError(f"Empty shard list file: {spec}")
        return out
    if any(ch in spec for ch in ["*", "?", "[", "]"]):
        out = sorted(glob.glob(spec))
        if not out:
            raise FileNotFoundError(f"No shards matched glob: {spec}")
        return out
    out = list(wds.shardlists.expand_urls(spec))
    if not out:
        raise FileNotFoundError(f"No shards matched spec: {spec}")
    return out


def _pick_device(device: str) -> torch.device:
    device = str(device).strip().lower()
    if device in {"auto", ""}:
        # Prefer CPU for broad compatibility.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device in {"cpu", "cuda", "npu"}:
        return torch.device(device)
    raise ValueError(f"Unknown device: {device}")


def _normalize_caption(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray, memoryview)):
        x = x.decode("utf-8", errors="ignore")
    s = str(x)
    return " ".join(s.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def _load_npy_field(x: Any) -> np.ndarray:
    if x is None:
        raise KeyError("Missing npy field")
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (bytes, bytearray, memoryview)):
        return np.load(io.BytesIO(x))
    return np.asarray(x)


def _latent_to_tokens(lat: np.ndarray) -> torch.Tensor:
    """
    Convert dataset latent to flattened token layout [N,4] in *scaled latent* space.
    Accepts:
      - [4,H,W]
      - [H,W,4]
      - [N,4]
    """
    arr = np.asarray(lat)
    if arr.ndim == 3:
        if arr.shape[0] == 4:
            # [4,H,W] -> [H,W,4] -> [N,4]
            return torch.from_numpy(arr).permute(1, 2, 0).reshape(-1, 4).to(dtype=torch.float32)
        if arr.shape[-1] == 4:
            return torch.from_numpy(arr).reshape(-1, 4).to(dtype=torch.float32)
        raise ValueError(f"Unrecognized latent shape (ndim=3): {tuple(arr.shape)}")
    if arr.ndim == 2 and arr.shape[-1] == 4:
        return torch.from_numpy(arr).to(dtype=torch.float32)
    raise ValueError(f"Unrecognized latent ndim={arr.ndim} shape={tuple(arr.shape)}")


def _tokens_to_lat_chw(tokens: torch.Tensor, *, latent_h: int, latent_w: int) -> torch.Tensor:
    """
    [N,4] -> [1,4,H,W]
    """
    tokens = tokens.to(dtype=torch.float32)
    n, d = tokens.shape
    if d != 4:
        raise ValueError(f"Expected dim_latent=4, got tokens shape {tuple(tokens.shape)}")
    h, w = int(latent_h), int(latent_w)
    if h <= 0 or w <= 0:
        side = int(math.isqrt(int(n)))
        if side * side != int(n):
            raise ValueError(f"Cannot infer square latent shape from N={n}; pass --latent_h/--latent_w")
        h, w = side, side
    if h * w != int(n):
        raise ValueError(f"latent_h*latent_w must equal N={n}, got {h}*{w}")
    lat = tokens.reshape(h, w, d).permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]
    return lat


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


@dataclass
class Args:
    model_dir: str = ""
    wds_shards: str = ""
    output_dir: str = "data/vis/overfit_eval"

    # Which training sample to evaluate
    sample_index: int = 0
    sample_key: str | None = None

    # VAE + latent convention
    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215
    latent_h: int = 16
    latent_w: int = 16

    # prompt building
    use_meta_input_ids: bool = True  # if json.input_ids exists, use it directly
    max_caption_tokens: int = 128  # used only when meta ids are not available

    # sampling config
    dt: float = 0.1
    max_steps: int = 50
    temperature: float = 0.0
    use_pi_gate: bool = True
    edit_prompt: bool = False
    image_num_tokens: int = 256
    seed: int = 42

    device: str = "cpu"  # cpu | cuda | npu | auto
    vae_device: str | None = None  # optional: separate device for VAE decode (e.g. cpu)


def main() -> None:
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")
    if not args.wds_shards:
        raise ValueError("--wds_shards is required")

    model_dir = os.path.abspath(os.path.expanduser(str(args.model_dir)))
    if not os.path.isdir(model_dir):
        parent = os.path.dirname(model_dir)
        hint = ""
        if os.path.isdir(parent):
            cands = sorted(glob.glob(os.path.join(parent, "checkpoint-*")))
            if cands:
                shown = ", ".join(os.path.basename(p) for p in cands[:20])
                hint = f"\nAvailable checkpoints under {parent}: {shown}"
        raise FileNotFoundError(f"--model_dir is not a directory: {args.model_dir} (abs: {model_dir}){hint}")

    shards = expand_shards(args.wds_shards)
    os.makedirs(args.output_dir, exist_ok=True)

    device = _pick_device(args.device)
    vae_device = device if args.vae_device is None else _pick_device(args.vae_device)
    transformers.set_seed(int(args.seed))

    print(f"[overfit_eval_wds] model_dir={model_dir}")
    print(f"[overfit_eval_wds] wds_shards={args.wds_shards} (resolved {len(shards)} tars)")
    print(f"[overfit_eval_wds] output_dir={args.output_dir} device={device} vae_device={vae_device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = OneFlowModel.from_pretrained(model_dir, map_location="cpu").eval().to(device)
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

    # Load one sample from dataset
    ds = wds.WebDataset(
        shards,
        shardshuffle=0,
        handler=wds.warn_and_continue,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    chosen = None
    cur = 0
    for sample in ds:
        key = str(sample.get("__key__", ""))
        if args.sample_key is not None:
            if key == str(args.sample_key):
                chosen = sample
                break
        else:
            if cur == int(args.sample_index):
                chosen = sample
                break
            cur += 1

    if chosen is None:
        raise ValueError(
            f"Could not find sample (sample_key={args.sample_key!r}, sample_index={args.sample_index}) "
            f"in shards={args.wds_shards}"
        )

    key = str(chosen.get("__key__", f"idx{args.sample_index}"))
    caption = _normalize_caption(chosen.get("txt", ""))
    lat_np = _load_npy_field(chosen.get("npy", None))
    gt_tokens = _latent_to_tokens(lat_np)  # [N,4] scaled

    # prompt ids (prefer meta input_ids for exact train/eval match)
    prompt_ids: list[int]
    meta = chosen.get("json", None)
    meta_dict = None
    if isinstance(meta, (bytes, bytearray, memoryview)):
        try:
            meta_dict = json.loads(meta)
        except Exception:
            meta_dict = None
    elif isinstance(meta, dict):
        meta_dict = meta

    if bool(args.use_meta_input_ids) and isinstance(meta_dict, dict) and "input_ids" in meta_dict:
        x1 = meta_dict["input_ids"]
        if not isinstance(x1, list) or not x1:
            raise ValueError("meta['input_ids'] exists but is not a non-empty list")
        prompt_ids = [int(t) for t in x1]
        prompt_source = "meta.input_ids"
    else:
        # Build exactly like training fallback: [BOS] + caption + [IMAGE] + [EOS]
        ids = tokenizer.encode(caption, add_special_tokens=False)[: int(args.max_caption_tokens)]
        bos = int(tokenizer.bos_token_id)
        eos = int(tokenizer.eos_token_id)
        image_token_id = int(tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
        prompt_ids = [bos] + [int(t) for t in ids] + [image_token_id] + [eos]
        prompt_source = "tokenize(caption)"

    n_img_tokens = sum(1 for t in prompt_ids if int(t) == int(tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN)))
    if n_img_tokens != 1:
        print(f"[warn] prompt contains {n_img_tokens} image tokens; this tool will compare only the first image.")

    # Run sampler
    out = sampler.sample([prompt_ids], cfg, return_dict=True)
    assert isinstance(out, OneFlowSamplerOutput)
    if not out.images:
        raise RuntimeError("Sampler produced no images; check that prompt contains <|oneflow_image|>.")

    gen_tokens = out.images[0].detach().to("cpu", dtype=torch.float32)  # [N,4] scaled
    gt_tokens = gt_tokens.detach().to("cpu", dtype=torch.float32)

    # Compare in latent space (scaled)
    if gen_tokens.shape != gt_tokens.shape:
        print(f"[warn] token shape mismatch: gen={tuple(gen_tokens.shape)} gt={tuple(gt_tokens.shape)}")
    n = min(int(gen_tokens.shape[0]), int(gt_tokens.shape[0]))
    gen_cmp = gen_tokens[:n]
    gt_cmp = gt_tokens[:n]
    latent_mse = float(((gen_cmp - gt_cmp) ** 2).mean().item())

    # Decode both to pixels
    print("[overfit_eval_wds] Loading VAE:", args.vae_id_or_path)
    vae = AutoencoderKL.from_pretrained(args.vae_id_or_path).to(vae_device).eval()

    gt_lat = _tokens_to_lat_chw(gt_tokens, latent_h=int(args.latent_h), latent_w=int(args.latent_w)).to(vae_device)
    gen_lat = _tokens_to_lat_chw(gen_tokens, latent_h=int(args.latent_h), latent_w=int(args.latent_w)).to(vae_device)
    gt_lat = gt_lat / float(args.latent_scale)
    gen_lat = gen_lat / float(args.latent_scale)

    with torch.no_grad():
        gt_img = vae.decode(gt_lat).sample
        gen_img = vae.decode(gen_lat).sample
        gt_img = (gt_img / 2 + 0.5).clamp(0, 1)
        gen_img = (gen_img / 2 + 0.5).clamp(0, 1)

    pixel_mse = float(((gt_img - gen_img) ** 2).mean().item())
    pixel_psnr = _psnr_from_mse(pixel_mse)

    # Save artifacts
    gt_path = os.path.join(args.output_dir, f"gt_{key}.png")
    gen_path = os.path.join(args.output_dir, f"gen_{key}.png")
    diff_path = os.path.join(args.output_dir, f"diff_{key}.png")
    save_image(gt_img.detach().cpu(), gt_path)
    save_image(gen_img.detach().cpu(), gen_path)
    save_image((gt_img - gen_img).abs().detach().cpu(), diff_path)

    text_path = os.path.join(args.output_dir, f"caption_{key}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
        f.write(f"\n[prompt_source] {prompt_source}\n")
        f.write(f"[prompt_len] {len(prompt_ids)}\n")

    seq_text = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=False)
    gen_text_path = os.path.join(args.output_dir, f"gen_text_{key}.txt")
    with open(gen_text_path, "w", encoding="utf-8") as f:
        f.write(seq_text + "\n")

    metrics = {
        "key": key,
        "prompt_source": prompt_source,
        "latent_tokens_gen": list(gen_tokens.shape),
        "latent_tokens_gt": list(gt_tokens.shape),
        "latent_mse": latent_mse,
        "pixel_mse": pixel_mse,
        "pixel_psnr": pixel_psnr,
        "paths": {
            "gt": os.path.basename(gt_path),
            "gen": os.path.basename(gen_path),
            "diff": os.path.basename(diff_path),
            "caption": os.path.basename(text_path),
            "gen_text": os.path.basename(gen_text_path),
        },
    }
    metrics_path = os.path.join(args.output_dir, f"metrics_{key}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Overfit eval ===")
    print("key:", key)
    print("caption:", caption)
    print("prompt_source:", prompt_source)
    print("latent_mse:", latent_mse)
    print("pixel_mse:", pixel_mse)
    print("pixel_psnr:", pixel_psnr)
    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()


