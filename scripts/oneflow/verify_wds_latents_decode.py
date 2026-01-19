#!/usr/bin/env python3
"""
Verify that WebDataset *latents* shards can be decoded back to images.

This answers a critical debugging question:
  "Is my dataset itself decodable (and does it look like what the caption says)?"

It reads samples from tar shards produced by `scripts/oneflow/precompute_latents_wds.py`:
  - `npy`: latent array (usually scaled by latent_scale, default 0.18215)
  - `txt`: caption
  - optional `json`: metadata (e.g., input_ids)

Then it decodes the latents using an SD VAE (`stabilityai/sd-vae-ft-mse`) and writes PNGs.

Example:
  python -u scripts/oneflow/verify_wds_latents_decode.py \
    --shards "data/latents_128_bundle/wds_latents_4096" \
    --vae_id_or_path "stabilityai/sd-vae-ft-mse" \
    --output_dir "data/vis/gt_decode_check" \
    --max_samples 16 \
    --latent_h 16 --latent_w 16
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _read_text_lines(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def expand_shards(spec: str) -> list[str]:
    spec = str(spec)
    if os.path.isdir(spec):
        out = [os.path.join(spec, n) for n in sorted(os.listdir(spec)) if n.endswith(".tar")]
        if not out:
            raise FileNotFoundError(f"No .tar shards found in dir: {spec}")
        return out

    if os.path.isfile(spec) and spec.endswith(".tar"):
        return [spec]

    if os.path.isfile(spec) and spec.endswith(".txt"):
        out = _read_text_lines(spec)
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
    import webdataset as wds

    out = list(wds.shardlists.expand_urls(spec))
    if not out:
        raise FileNotFoundError(f"No shards matched spec: {spec}")
    return out


def _pick_device(device: str) -> torch.device:
    import torch

    device = str(device).strip().lower()
    if device in {"auto", ""}:
        # Keep CPU default for maximum compatibility (diffusers+NPU can be finicky).
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device in {"npu"}:
        return torch.device("npu")
    if device in {"cuda", "cpu"}:
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
    # webdataset may already decode into something array-like
    return np.asarray(x)


def _latent_to_chw(lat: np.ndarray, *, latent_h: int, latent_w: int) -> tuple[torch.Tensor, int, int]:
    """
    Convert a latent array into a torch tensor [1,4,H,W] in *scaled latent* space.
    Supports lat shapes:
      - [4,H,W]
      - [H,W,4]
      - [N,4] (needs latent_h/latent_w or infer square)
    Returns: (lat_chw, H, W)
    """
    import torch

    arr = np.asarray(lat)
    if arr.ndim == 3:
        if arr.shape[0] == 4:
            h, w = int(arr.shape[1]), int(arr.shape[2])
            t = torch.from_numpy(arr).to(dtype=torch.float32).unsqueeze(0)  # [1,4,H,W]
            return t, h, w
        if arr.shape[-1] == 4:
            h, w = int(arr.shape[0]), int(arr.shape[1])
            t = torch.from_numpy(arr).to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]
            return t, h, w
        raise ValueError(f"Unrecognized latent shape (ndim=3): {tuple(arr.shape)}")

    if arr.ndim == 2 and arr.shape[-1] == 4:
        n = int(arr.shape[0])
        if latent_h > 0 and latent_w > 0:
            h, w = int(latent_h), int(latent_w)
            if h * w != n:
                raise ValueError(f"latent_h*latent_w must equal N={n}, got {h}*{w}")
        else:
            side = int(math.isqrt(n))
            if side * side != n:
                raise ValueError(f"Cannot infer square latent shape from N={n}; pass --latent_h/--latent_w")
            h, w = side, side
        t = torch.from_numpy(arr).to(dtype=torch.float32).reshape(h, w, 4).permute(2, 0, 1).unsqueeze(0)
        return t, h, w

    raise ValueError(f"Unrecognized latent ndim={arr.ndim} shape={tuple(arr.shape)}")


@dataclass
class Args:
    shards: str = ""
    output_dir: str = "data/vis/gt_decode_check"

    vae_id_or_path: str = "stabilityai/sd-vae-ft-mse"
    latent_scale: float = 0.18215

    # latent shape (optional; only needed when npy is stored as [N,4])
    latent_h: int = 0
    latent_w: int = 0

    max_samples: int = 32
    device: str = "cpu"  # cpu | cuda | npu | auto
    local_files_only: bool = False

    # keys inside each wds sample
    latent_key: str = "npy"
    caption_key: str = "txt"

    # Optional: decode only these keys (useful for debugging specific samples)
    sample_key: list[str] | None = None
    sample_keys_file: str | None = None

    # Error handling
    fail_fast: bool = False
    max_failures: int = 100
    overwrite: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Verify that WDS latents can be decoded by an SD VAE.")
    p.add_argument("--shards", required=True, help="Shard spec: dir/glob/brace/list.txt/or .tar file.")
    p.add_argument("--output_dir", default="data/vis/gt_decode_check")

    p.add_argument("--vae_id_or_path", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--local_files_only", action="store_true", help="Load VAE from local cache only (offline).")
    p.add_argument("--latent_scale", type=float, default=0.18215)

    p.add_argument("--latent_h", type=int, default=0, help="Needed if latents are stored as [N,4].")
    p.add_argument("--latent_w", type=int, default=0, help="Needed if latents are stored as [N,4].")

    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--device", default="cpu", help="cpu | cuda | npu | auto")

    p.add_argument("--latent_key", default="npy", help="Field name for latents inside each sample.")
    p.add_argument("--caption_key", default="txt", help="Field name for caption inside each sample.")

    p.add_argument(
        "--sample_key",
        action="append",
        default=None,
        help="Decode only this __key__ (repeatable). Example: --sample_key 000070352",
    )
    p.add_argument(
        "--sample_keys_file",
        default=None,
        help="Path to a text file with one sample __key__ per line.",
    )

    p.add_argument("--fail_fast", action="store_true", help="Stop on first failure.")
    p.add_argument("--max_failures", type=int, default=100, help="Stop after this many failures.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing png/txt outputs.")

    ns = p.parse_args()
    return Args(
        shards=ns.shards,
        output_dir=ns.output_dir,
        vae_id_or_path=ns.vae_id_or_path,
        latent_scale=float(ns.latent_scale),
        latent_h=int(ns.latent_h),
        latent_w=int(ns.latent_w),
        max_samples=int(ns.max_samples),
        device=str(ns.device),
        local_files_only=bool(ns.local_files_only),
        latent_key=str(ns.latent_key),
        caption_key=str(ns.caption_key),
        sample_key=list(ns.sample_key) if ns.sample_key else None,
        sample_keys_file=str(ns.sample_keys_file) if ns.sample_keys_file else None,
        fail_fast=bool(ns.fail_fast),
        max_failures=int(ns.max_failures),
        overwrite=bool(ns.overwrite),
    )


_SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_stem(x: str) -> str:
    x = str(x)
    x = x.replace(os.sep, "_")
    x = _SAFE_STEM_RE.sub("_", x)
    return x.strip("_")[:200] or "sample"


def _disable_torch_npu_detection_for_cpu() -> None:
    """
    In some environments, `torch_npu` is installed but not usable (e.g., missing libhccl.so).
    Diffusers + Transformers may try to import torch_npu just because it's installed, causing crashes.

    This function patches their NPU availability checks *before* importing diffusers models.
    It is only used for non-NPU runs.
    """
    try:
        import transformers.utils.import_utils as tiu

        tiu.is_torch_npu_available = lambda check_device=False: False  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import diffusers.utils.import_utils as diu

        diu._torch_npu_available = False  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    if not args.shards:
        raise ValueError("--shards is required (dir/glob/brace/txtlist)")

    # ---- environment guardrails -------------------------------------------------
    # - For CPU/CUDA runs: disable backend extension auto-loading (torch can crash trying to load torch_npu).
    # - For NPU runs: expect user to have sourced Ascend env so libhccl/libascend are visible.
    dev_req = str(args.device).strip().lower()
    if dev_req not in {"npu"}:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch
    import webdataset as wds
    from torchvision.utils import save_image

    # If not explicitly running on NPU, and torch_npu is not usable, patch diffusers/transformers to ignore it.
    if dev_req not in {"npu"}:
        try:
            import torch_npu  # noqa: F401
        except Exception:
            _disable_torch_npu_detection_for_cpu()

    # Diffusers import must happen after the patch above.
    from diffusers.models import AutoencoderKL

    shards = expand_shards(args.shards)
    os.makedirs(args.output_dir, exist_ok=True)

    device = _pick_device(args.device)
    print(f"[verify_wds_latents_decode] shards={args.shards} (resolved {len(shards)} tars)")
    print(f"[verify_wds_latents_decode] output_dir={args.output_dir}")
    print(f"[verify_wds_latents_decode] device={device} vae={args.vae_id_or_path} latent_scale={args.latent_scale}")
    print(f"[verify_wds_latents_decode] latent_key={args.latent_key} caption_key={args.caption_key}")

    print("[verify_wds_latents_decode] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_id_or_path, local_files_only=bool(args.local_files_only)).to(device).eval()

    key_allow: set[str] | None = None
    keys: list[str] = []
    if args.sample_key:
        keys.extend([str(k).strip() for k in args.sample_key if str(k).strip()])
    if args.sample_keys_file:
        keys.extend(_read_text_lines(args.sample_keys_file))
    if keys:
        key_allow = set(keys)
        print(f"[verify_wds_latents_decode] filtering: {len(key_allow)} keys")

    # Disable distributed splitting even if RANK/WORLD_SIZE env vars are present.
    ds = wds.WebDataset(
        shards,
        shardshuffle=0,
        handler=wds.warn_and_continue,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    written = 0
    failed = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for sample in ds:
            if written >= int(args.max_samples):
                break
            try:
                key = sample.get("__key__", f"idx{written}")
                if key_allow is not None and str(key) not in key_allow:
                    continue
                caption = _normalize_caption(sample.get(str(args.caption_key), ""))
                lat_np = _load_npy_field(sample.get(str(args.latent_key), None))

                lat_chw, h, w = _latent_to_chw(lat_np, latent_h=int(args.latent_h), latent_w=int(args.latent_w))
                lat_chw = lat_chw.to(device=device, dtype=torch.float32)
                lat_chw = lat_chw / float(args.latent_scale)
                if not torch.isfinite(lat_chw).all():
                    raise ValueError("latent contains NaN/Inf")

                with torch.no_grad():
                    img = vae.decode(lat_chw).sample  # [-1,1]
                    img = (img / 2 + 0.5).clamp(0, 1)
                    if not torch.isfinite(img).all():
                        raise ValueError("decoded image contains NaN/Inf")

                stem = f"{written:05d}_{_safe_stem(str(key))}"
                png_path = os.path.join(args.output_dir, f"{stem}.png")
                txt_path = os.path.join(args.output_dir, f"{stem}.txt")
                if (not bool(args.overwrite)) and (os.path.exists(png_path) or os.path.exists(txt_path)):
                    print(f"[skip] exists: {png_path}")
                else:
                    save_image(img.detach().cpu(), png_path)
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(caption + "\n")

                rec = {
                    "index": written,
                    "key": str(key),
                    "caption": caption,
                    "latent_shape": list(lat_np.shape),
                    "decoded_hw": [h, w],
                    "latent_scale": float(args.latent_scale),
                    "png": os.path.basename(png_path),
                    "txt": os.path.basename(txt_path),
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                print(f"[{written}] key={key} latent={tuple(lat_np.shape)} -> {png_path}")
                written += 1
            except Exception as e:
                print(f"[warn] failed to decode sample (index={written}): {e}")
                failed += 1
                if bool(args.fail_fast) or failed >= int(args.max_failures):
                    raise
                continue

    print(f"[verify_wds_latents_decode] done. wrote={written} manifest={manifest_path}")


if __name__ == "__main__":
    main()


