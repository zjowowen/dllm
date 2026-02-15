#!/usr/bin/env python3
"""
Check whether a local FineWeb path is directly usable for offline training.

This script is read-only: it does not modify any file. It classifies a path into:
1) `save_to_disk` dataset directory
2) parquet directory
3) metadata-only cache (not trainable as-is)

Example:
  python -u scripts/oneflow/check_fineweb_local_readiness.py \
    --fineweb_root /mnt/shared-storage-user/ai4sreason/zhangjinouwen/huggingface/fineweb-edu_sample-10BT \
    --output_json /tmp/fineweb_readiness.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ReadinessResult:
    fineweb_root: str
    exists: bool
    ready: bool
    ready_mode: str
    save_to_disk_candidates: list[str]
    parquet_files_count: int
    parquet_example: list[str]
    metadata_files_count: int
    metadata_example: list[str]
    notes: list[str]
    next_steps: list[str]
    path_conventions: dict[str, str]


def _is_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "dataset_dict.json").exists()


def _scan_dataset_candidates(root: Path, max_depth: int = 4) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    root_depth = len(root.parts)
    for curr, dirnames, _ in os.walk(root):
        p = Path(curr)
        depth = len(p.parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        if _is_dataset_dir(p):
            out.append(p)
    return out


def _scan_files(root: Path, suffix: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(f"*{suffix}"))


def build_result(root: Path) -> ReadinessResult:
    exists = root.exists()
    candidates = _scan_dataset_candidates(root)
    parquet_files = _scan_files(root, ".parquet")
    metadata_files = _scan_files(root, ".parquet.metadata")

    ready_mode = "none"
    ready = False
    notes: list[str] = []
    next_steps: list[str] = []

    if candidates:
        ready_mode = "save_to_disk"
        ready = True
        notes.append("Detected directory that looks like a datasets.save_to_disk export.")
        next_steps.extend(
            [
                "Use this path as PT input: --dataset_args <candidate_dir> --load_preprocessed_data True",
                "Ensure tokenizer path is paired: <bundle>/tokenizer",
            ]
        )
    elif parquet_files:
        ready_mode = "parquet"
        ready = True
        notes.append("Detected parquet files; path can be used for conversion to offline PT bundle.")
        next_steps.extend(
            [
                "Convert parquet path into PT bundle via scripts/oneflow/prepare_pt_text_dataset.py",
                "Then train with --dataset_args <bundle>/dataset --load_preprocessed_data True",
            ]
        )
    elif metadata_files:
        ready_mode = "metadata_only"
        ready = False
        notes.append("Only '.parquet.metadata' files were found; no trainable dataset files detected.")
        next_steps.extend(
            [
                "Populate local parquet snapshot first, OR generate a save_to_disk dataset bundle.",
                "If this is HF cache-only metadata, point to the real parquet root instead of cache metadata.",
            ]
        )
    else:
        ready_mode = "none"
        ready = False
        notes.append("No save_to_disk markers and no parquet files detected under the path.")
        next_steps.append("Check the mount path and re-run this checker on the actual dataset location.")

    path_conventions = {
        "PT_BUNDLE": "<workspace>/data/offline/pt_text_fineweb_edu_<tag>",
        "SFT_BUNDLE": "<workspace>/data/offline/sft_text_pseudo_from_pt_<tag>",
        "TOKENIZER_DIR": "$PT_BUNDLE/tokenizer",
        "CKPT_DIR": "<workspace>/data/ckpts/oneflow_text_<tag>",
    }

    if ready_mode == "metadata_only":
        next_steps.extend(
            [
                "Recommended conversion entry: scripts/oneflow/prepare_pt_text_dataset.py",
                "Recommended launch entry: scripts/oneflow/launch_pt_text_h200.sh (to be added in this branch).",
            ]
        )

    return ReadinessResult(
        fineweb_root=str(root),
        exists=exists,
        ready=ready,
        ready_mode=ready_mode,
        save_to_disk_candidates=[str(p) for p in candidates[:10]],
        parquet_files_count=len(parquet_files),
        parquet_example=[str(p) for p in parquet_files[:5]],
        metadata_files_count=len(metadata_files),
        metadata_example=[str(p) for p in metadata_files[:5]],
        notes=notes,
        next_steps=next_steps,
        path_conventions=path_conventions,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fineweb_root",
        type=str,
        default="/mnt/shared-storage-user/ai4sreason/zhangjinouwen/huggingface/fineweb-edu_sample-10BT",
        help="Root path to inspect.",
    )
    p.add_argument(
        "--require_ready",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=False,
        help="If true, return non-zero when path is not train-ready.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional JSON output path.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(os.path.expanduser(args.fineweb_root))
    result = build_result(root)
    payload = asdict(result)

    print(json.dumps(payload, ensure_ascii=True, indent=2))

    if args.output_json:
        out = Path(os.path.expanduser(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[INFO] wrote report: {out}")

    if bool(args.require_ready) and (not result.ready):
        print("[ERROR] dataset path is not ready for offline training.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
