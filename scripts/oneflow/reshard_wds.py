#!/usr/bin/env python3
"""
Reshard an existing WebDataset (tar shards) into a larger number of smaller shards.

Why:
  When training with distributed workers (e.g., 16 NPUs), it's recommended to have
  at least as many tar shards as the world size; otherwise some ranks may get an
  empty shard list (or you end up duplicating shards across ranks).

This script only re-packs the existing samples; it does NOT recompute VAE latents.

Example:
  python -u scripts/oneflow/reshard_wds.py \
    --input_shards "data/latents_128_bundle/wds_latents" \
    --output_dir "data/latents_128_bundle/wds_latents_4096" \
    --maxcount 4096

Then train with:
  --shards "data/latents_128_bundle/wds_latents_4096"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any

import webdataset as wds


def expand_shards(spec: str) -> list[str]:
    """
    Expand shard spec into a list of local shard paths.
    Supports:
      - a directory containing *.tar
      - a text file listing shards (one per line)
      - glob patterns (e.g., "/path/*.tar")
      - brace expansion (e.g., "/path/shard-{000000..000099}.tar")
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


def main() -> int:
    p = argparse.ArgumentParser(description="Reshard WebDataset tar shards into smaller shards.")
    p.add_argument("--input_shards", required=True, help="Input shard spec (dir/glob/brace/txtlist).")
    p.add_argument("--output_dir", required=True, help="Output directory to write new tar shards.")
    p.add_argument("--maxcount", type=int, default=4096, help="Samples per output shard.")
    p.add_argument("--shard_pattern", default="shard-%06d.tar", help="Output tar naming pattern.")
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap for debugging.")
    p.add_argument(
        "--caption_key",
        default="txt",
        help="Caption field name used for filtering (default: txt).",
    )
    p.add_argument(
        "--caption_contains",
        default=None,
        help="Optional case-insensitive substring filter on caption (e.g., flower).",
    )
    p.add_argument(
        "--caption_regex",
        default=None,
        help=r"Optional regex filter on caption (Python re, case-insensitive). Example: '\bflower\b'",
    )
    args = p.parse_args()

    in_shards = expand_shards(args.input_shards)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[reshard_wds] input_shards={args.input_shards}")
    print(f"[reshard_wds] num_input_shards={len(in_shards)}")
    print(f"[reshard_wds] output_dir={args.output_dir}")
    print(f"[reshard_wds] maxcount={int(args.maxcount)} shard_pattern={args.shard_pattern}")
    if args.caption_contains or args.caption_regex:
        print(
            f"[reshard_wds] filter: caption_key={args.caption_key} "
            f"caption_contains={args.caption_contains!r} caption_regex={args.caption_regex!r}"
        )

    ds = wds.WebDataset(in_shards, shardshuffle=0, handler=wds.warn_and_continue)
    writer = wds.ShardWriter(os.path.join(args.output_dir, args.shard_pattern), maxcount=int(args.maxcount))

    total = 0  # written
    seen = 0
    cap_contains = str(args.caption_contains).lower() if args.caption_contains else None
    cap_re = re.compile(str(args.caption_regex), flags=re.IGNORECASE) if args.caption_regex else None
    for sample in ds:
        seen += 1
        if args.max_samples is not None and total >= int(args.max_samples):
            break
        if not isinstance(sample, dict):
            continue
        key = sample.get("__key__", None)
        if key is None:
            continue

        # Optional caption-based filtering (useful for making a tiny targeted dataset)
        if cap_contains is not None or cap_re is not None:
            cap = sample.get(str(args.caption_key), None)
            if cap is None:
                continue
            if isinstance(cap, (bytes, bytearray, memoryview)):
                cap = cap.decode("utf-8", errors="ignore")
            cap_s = str(cap)
            if cap_contains is not None and cap_contains not in cap_s.lower():
                continue
            if cap_re is not None and cap_re.search(cap_s) is None:
                continue

        out: dict[str, Any] = {"__key__": str(key)}
        for k, v in sample.items():
            # skip webdataset internal keys
            if str(k).startswith("__"):
                continue
            out[str(k)] = v

        writer.write(out)
        total += 1
        if total > 0 and total % 100000 == 0:
            print(f"[reshard_wds] written={total}")

    writer.close()
    print(f"[reshard_wds] done. written={total} (seen={seen})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


