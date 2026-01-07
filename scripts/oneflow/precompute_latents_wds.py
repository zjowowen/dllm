"""
Precompute VAE latents for an image-caption WebDataset (tar shards).

This is designed for **offline cluster training**:
1) On a machine WITH internet (or access to raw data), prepare image shards (e.g., with img2dataset).
2) Run this script to encode images into SD-VAE latents and write a new WebDataset containing:
   - `npy`: latent array, shape [4, H/8, W/8] (float16 by default)
   - `txt`: caption text (utf-8)
   - optional `json`: metadata (e.g., precomputed `input_ids`)
3) Copy the output folder to the no-network cluster.

Example:
  python -u scripts/oneflow/precompute_latents_wds.py \
    --input_shards "/data/cc3m_wds_128/shard-{000000..000099}.tar" \
    --output_dir "/data/cc3m_latents_128_bundle" \
    --image_size 128 \
    --vae_id_or_path "stabilityai/sd-vae-ft-mse" \
    --batch_size 64 \
    --num_workers 8 \
    --maxcount 10000 \
    --tokenizer_name_or_path "gpt2" \
    --max_caption_tokens 128 \
    --write_input_ids True

Outputs:
  <output_dir>/
    wds_latents/         (tar shards with npy/txt/json)
    tokenizer/           (saved tokenizer, if tokenizer_name_or_path is set)
    stats.json           (dataset stats & config)

Notes:
  - Latents are scaled by `latent_scale` (default 0.18215) to match Stable Diffusion conventions.
  - For determinism, use `--latent_mode mean`. Default is `sample`.
"""

from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers
import webdataset as wds
from diffusers.models import AutoencoderKL
from torchvision import transforms as T

from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_EOM, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_TOKEN
from dllm.utils.utils import get_default_logger

logger = get_default_logger(__name__)


@dataclass
class Args:
    # Input WebDataset shards (from img2dataset or your own tar shards)
    input_shards: str = ""
    # Output bundle dir
    output_dir: str = ""

    # Input keys
    image_keys: str = "jpg,png,webp"
    caption_key: str = "txt"  # usually `txt` for img2dataset webdataset output

    # Image preprocessing
    image_size: int = 128
    center_crop: bool = True

    # VAE
    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215
    latent_mode: str = "sample"  # "sample" | "mean"
    dtype: str = "fp16"  # "fp16" | "bf16" | "fp32"

    # Encoding loop
    batch_size: int = 64
    num_workers: int = 8
    max_samples: int | None = None
    # WebDataset expects an int (e.g., 1000) or 0/False
    shardshuffle: int = 0

    # Output shards
    maxcount: int = 10000  # samples per shard
    shard_pattern: str = "shard-%06d.tar"
    save_latents_dtype: str = "fp16"  # "fp16" | "fp32"

    # Optional: precompute input_ids and save tokenizer (recommended for offline cluster)
    tokenizer_name_or_path: str | None = None
    max_caption_tokens: int = 128
    write_input_ids: bool = True


def _dtype_from_str(s: str) -> torch.dtype:
    s = str(s).lower().strip()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def build_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizer:
    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if tok.eos_token is None:
        tok.eos_token = tok.pad_token
    if tok.bos_token is None:
        tok.bos_token = tok.pad_token
    tok.add_special_tokens(
        {"additional_special_tokens": [ONEFLOW_IMAGE_TOKEN, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_EOM]}
    )
    return tok


def expand_shards(spec: str) -> list[str]:
    """
    Expand shard spec into a list of local shard paths.
    Supports:
      - brace expansion: "/path/shard-{000000..000999}.tar"
      - glob patterns: "/path/shard-*.tar"
      - a directory containing *.tar
      - a text file listing shards (one per line)
    """
    spec = str(spec)
    if os.path.isdir(spec):
        out = []
        for name in sorted(os.listdir(spec)):
            if name.endswith(".tar"):
                out.append(os.path.join(spec, name))
        if not out:
            raise FileNotFoundError(f"No .tar shards found in dir: {spec}")
        return out

    if os.path.isfile(spec) and spec.endswith(".txt"):
        out = []
        with open(spec, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(line)
        if not out:
            raise FileNotFoundError(f"Empty shard list file: {spec}")
        return out

    # glob
    if any(ch in spec for ch in ["*", "?", "[", "]"]):
        out = sorted(glob.glob(spec))
        if not out:
            raise FileNotFoundError(f"No shards matched glob: {spec}")
        return out

    # braceexpand (webdataset helper)
    out = list(wds.shardlists.expand_urls(spec))
    if not out:
        raise FileNotFoundError(f"No shards matched spec: {spec}")
    return out


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.input_shards:
        raise ValueError("--input_shards is required")
    if not args.output_dir:
        raise ValueError("--output_dir is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_str(args.dtype)
    save_latents_dtype = _dtype_from_str(args.save_latents_dtype)

    out_wds_dir = os.path.join(args.output_dir, "wds_latents")
    os.makedirs(out_wds_dir, exist_ok=True)

    tok = None
    image_token_id: int | None = None
    bos_id: int | None = None
    eos_id: int | None = None
    if args.tokenizer_name_or_path:
        logger.info(f"Loading tokenizer: {args.tokenizer_name_or_path}")
        tok = build_tokenizer(args.tokenizer_name_or_path)
        tok_dir = os.path.join(args.output_dir, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok.save_pretrained(tok_dir)
        logger.info(f"Saved tokenizer to: {tok_dir}")

        image_token_id = int(tok.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
        if tok.unk_token_id is not None and image_token_id == int(tok.unk_token_id):
            raise ValueError(
                f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}. "
                "Please ensure the tokenizer is saved with OneFlow special tokens."
            )
        bos_id = int(tok.bos_token_id)
        eos_id = int(tok.eos_token_id)

    logger.info(f"Loading VAE: {args.vae_id_or_path}")
    vae = AutoencoderKL.from_pretrained(args.vae_id_or_path).to(device).eval()

    transform = T.Compose(
        [
            T.Resize(args.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(args.image_size) if args.center_crop else T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
        ]
    )

    image_keys = [k.strip() for k in str(args.image_keys).split(",") if k.strip()]
    caption_key = str(args.caption_key).strip()

    def map_sample(sample: dict[str, Any]) -> dict[str, Any]:
        # Expect PIL image via .decode("pil")
        key = sample["__key__"]

        pil = None
        for k in image_keys:
            if k in sample:
                pil = sample[k]
                break
        if pil is None:
            raise KeyError(f"Missing image key, tried={image_keys}, got={list(sample.keys())[:20]}")
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        img = transform(pil)  # [3,H,W] in [0,1]

        cap = sample.get(caption_key, None)
        if cap is None:
            raise KeyError(f"Missing caption key '{caption_key}' in sample keys={list(sample.keys())[:20]}")
        if isinstance(cap, (bytes, bytearray)):
            cap = cap.decode("utf-8", errors="ignore")
        cap = str(cap).strip()
        if not cap:
            raise ValueError("Empty caption")

        return {"key": str(key), "image": img, "caption": cap}

    shards = expand_shards(args.input_shards)
    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=int(args.shardshuffle),
            handler=wds.warn_and_continue,
        )
        .decode("pil")
        .map(map_sample, handler=wds.warn_and_continue)
    )

    loader = wds.WebLoader(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle=False,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # shard writer
    writer = wds.ShardWriter(os.path.join(out_wds_dir, args.shard_pattern), maxcount=int(args.maxcount))

    total = 0
    first_latent_shape: tuple[int, ...] | None = None

    autocast = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else None

    logger.info("Start encoding latents...")
    for batch in loader:
        if args.max_samples is not None and total >= int(args.max_samples):
            break

        images = batch["image"].to(device=device, dtype=torch.float32)  # encode in fp32 for stability
        captions = batch["caption"]
        keys = batch["key"]

        # Encode
        with torch.no_grad():
            if autocast is None:
                enc = vae.encode(images * 2 - 1).latent_dist
            else:
                with autocast:
                    enc = vae.encode(images * 2 - 1).latent_dist

            if str(args.latent_mode).lower().strip() == "mean":
                lat = enc.mean
            else:
                lat = enc.sample()

            lat = float(args.latent_scale) * lat

        # Convert dtype for storage
        lat = lat.to(dtype=save_latents_dtype)
        lat_np = lat.detach().cpu().numpy()

        if first_latent_shape is None and lat_np.shape:
            first_latent_shape = tuple(lat_np.shape[1:])

        # Optional precomputed ids (compute in the main process; tokenizers + dataloader
        # workers can be fragile under fork/spawn).
        ids_list = None
        if tok is not None and bool(args.write_input_ids):
            ids_list = []
            for cap in captions:
                ids = tok.encode(str(cap), add_special_tokens=False)[: int(args.max_caption_tokens)]
                x1 = [int(bos_id)] + [int(t) for t in ids] + [int(image_token_id)] + [int(eos_id)]
                ids_list.append(x1)

        for i in range(lat_np.shape[0]):
            if args.max_samples is not None and total >= int(args.max_samples):
                break
            item = {
                "__key__": str(keys[i]),
                "npy": lat_np[i].astype(np.float16 if save_latents_dtype == torch.float16 else np.float32),
                "txt": str(captions[i]),
            }
            if ids_list is not None:
                item["json"] = {"input_ids": [int(x) for x in ids_list[i]]}
            writer.write(item)
            total += 1

        if total and total % 5000 == 0:
            logger.info(f"Encoded {total} samples...")

    writer.close()

    stats = {
        "num_samples": int(total),
        "image_size": int(args.image_size),
        "latent_scale": float(args.latent_scale),
        "latent_mode": str(args.latent_mode),
        "latent_shape": list(first_latent_shape) if first_latent_shape is not None else None,
        "caption_key": str(args.caption_key),
        "image_keys": image_keys,
        "tokenizer_saved": bool(args.tokenizer_name_or_path),
        "write_input_ids": bool(args.tokenizer_name_or_path) and bool(args.write_input_ids),
        "max_caption_tokens": int(args.max_caption_tokens),
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Done.")
    logger.info(f"Output shards: {out_wds_dir}")
    logger.info(f"Stats: {os.path.join(args.output_dir, 'stats.json')}")


if __name__ == "__main__":
    main()


