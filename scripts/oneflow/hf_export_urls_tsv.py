#!/usr/bin/env python3
"""
Export HuggingFace dataset metadata to a `urls.tsv` for img2dataset.

This script is intended to run on a machine WITH internet access.

Example (CC3M / Conceptual Captions 3M):
  python -u scripts/oneflow/hf_export_urls_tsv.py \
    --dataset conceptual_captions --config unlabeled --split train \
    --output_tsv /path/to/urls.tsv \
    --max_rows 100000

Then you can run img2dataset:
  img2dataset --url_list /path/to/urls.tsv --input_format tsv --url_col url --caption_col caption ...
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


URL_CANDIDATES = (
    "url",
    "URL",
    "image_url",
    "img_url",
    "imageURL",
    "original_url",
    "jpg",
    "jpeg",
    "png",
    "webp",
)

CAPTION_CANDIDATES = (
    "caption",
    "text",
    "TEXT",
    "prompt",
    "description",
    "title",
)

# Avoid torch importing NPU backend extensions (torch_npu) on machines without HCCL properly set up.
# `datasets` may import torch for formatting, which can crash otherwise.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _normalize_cell(value: Any) -> str:
    """Make a safe single-line TSV cell."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value if x is not None)
    s = str(value)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    # collapse repeated whitespace
    return " ".join(s.split())


def _get_by_path(example: Dict[str, Any], path: str) -> Any:
    """Supports 'a.b.c' access for nested dicts."""
    cur: Any = example
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _guess_key(example: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    # exact match first
    for c in candidates:
        if c in example:
            return c
    # case-insensitive fallback
    lower_map = {k.lower(): k for k in example.keys()}
    for c in candidates:
        k = lower_map.get(c.lower())
        if k is not None:
            return k
    return None


def _load_dataset_compat(
    dataset: str,
    config: Optional[str],
    *,
    split: str,
    streaming: bool,
    trust_remote_code: bool,
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
):
    from datasets import load_dataset  # local import so the script errors clearly if missing

    sig = inspect.signature(load_dataset)
    valid = set(sig.parameters.keys())

    kwargs: Dict[str, Any] = {}
    if "split" in valid:
        kwargs["split"] = split
    if "streaming" in valid:
        kwargs["streaming"] = streaming
    if trust_remote_code and "trust_remote_code" in valid:
        kwargs["trust_remote_code"] = True
    if revision is not None and "revision" in valid:
        kwargs["revision"] = revision
    if cache_dir is not None and "cache_dir" in valid:
        kwargs["cache_dir"] = cache_dir
    if token:
        if "token" in valid:
            kwargs["token"] = token
        elif "use_auth_token" in valid:
            kwargs["use_auth_token"] = token

    try:
        if config:
            return load_dataset(dataset, config, **kwargs)
        return load_dataset(dataset, **kwargs)
    except ValueError as e:
        # Backward/forward compatibility for common naming differences.
        # In some environments, Conceptual Captions config is named "unlabeled" (3M),
        # while docs/users may refer to it as "3m".
        if dataset in {"conceptual_captions"} and (config or "").lower() in {"3m", "cc3m"}:
            msg = str(e)
            if "BuilderConfig" in msg and "unlabeled" in msg:
                print(
                    "[hf_export_urls_tsv] NOTE: conceptual_captions config '3m' not found in this "
                    "datasets version; falling back to config 'unlabeled' (CC3M).",
                    file=sys.stderr,
                )
                return load_dataset(dataset, "unlabeled", **kwargs)
        raise


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export a HuggingFace dataset split to urls.tsv (columns: url, caption)."
    )
    p.add_argument("--dataset", required=True, help="HF dataset name, e.g. conceptual_captions")
    p.add_argument("--config", default=None, help="Optional dataset config/name, e.g. 3m")
    p.add_argument("--split", default="train", help="Split name, e.g. train/validation/test")
    p.add_argument("--output_tsv", required=True, help="Output TSV path (will be overwritten)")
    p.add_argument("--max_rows", type=int, default=None, help="Limit rows written (for quick tests)")

    p.add_argument("--streaming", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument(
        "--no_streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming (downloads the dataset).",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Allow remote dataset script execution if required by the dataset.",
    )
    p.add_argument("--revision", default=None, help="Dataset git revision/tag/commit")
    p.add_argument("--cache_dir", default=None, help="HF cache dir")
    p.add_argument("--token", default=None, help="HF token for gated datasets (or set HF_TOKEN)")

    p.add_argument(
        "--url_col",
        default=None,
        help="URL column name. Supports nested paths like 'image.url'. If omitted, auto-detects.",
    )
    p.add_argument(
        "--caption_col",
        default=None,
        help="Caption/text column name. If omitted, auto-detects.",
    )
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    ds = _load_dataset_compat(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=token,
    )

    it = iter(ds)
    try:
        first = next(it)
    except StopIteration:
        print("Dataset split is empty.", file=sys.stderr)
        return 2

    if not isinstance(first, dict):
        print(f"Expected dict examples, got: {type(first)}", file=sys.stderr)
        return 2

    url_key = args.url_col or _guess_key(first, URL_CANDIDATES)
    cap_key = args.caption_col or _guess_key(first, CAPTION_CANDIDATES)

    if not url_key:
        print(
            "Could not auto-detect url column. "
            f"Please pass --url_col. Available keys: {sorted(first.keys())}",
            file=sys.stderr,
        )
        return 2
    if not cap_key:
        print(
            "Could not auto-detect caption column. "
            f"Please pass --caption_col. Available keys: {sorted(first.keys())}",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[hf_export_urls_tsv] dataset={args.dataset} config={args.config} split={args.split}")
    print(f"[hf_export_urls_tsv] url_col={url_key} caption_col={cap_key}")
    print(f"[hf_export_urls_tsv] output_tsv={out_path}")

    def extract(example: Dict[str, Any], key: str) -> Any:
        if "." in key:
            return _get_by_path(example, key)
        v = example.get(key)
        if isinstance(v, dict):
            # common patterns for nested structures
            for kk in ("url", "URL", "image_url", "text", "caption", "TEXT", "description"):
                if kk in v:
                    return v[kk]
        return v

    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["url", "caption"])

        def maybe_write(example: Dict[str, Any]) -> None:
            nonlocal written
            url = _normalize_cell(extract(example, url_key))
            if not url:
                return
            caption = _normalize_cell(extract(example, cap_key))
            w.writerow([url, caption])
            written += 1

        maybe_write(first)
        if args.max_rows is None or written < args.max_rows:
            for ex in it:
                if not isinstance(ex, dict):
                    continue
                maybe_write(ex)
                if args.max_rows is not None and written >= args.max_rows:
                    break
                if written > 0 and written % 100000 == 0:
                    print(f"[hf_export_urls_tsv] written={written}", file=sys.stderr)

    print(f"[hf_export_urls_tsv] done. written={written}")
    return 0


if __name__ == "__main__":
    # NOTE: On some environments (notably certain torch/accelerator stacks),
    # Python interpreter finalization may crash after successful execution
    # (e.g. "Fatal Python error: PyGILState_Release ... finalizing").
    # The TSV file is already fully written at that point, so we use a hard
    # exit to avoid running problematic finalizers.
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


