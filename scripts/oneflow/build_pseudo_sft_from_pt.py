#!/usr/bin/env python3
"""
Build a pseudo-SFT offline bundle from a PT offline bundle.

Input:
  <pt_bundle>/dataset    (DatasetDict with `input_ids`)
  <pt_bundle>/tokenizer  (optional if --tokenizer_name_or_path is provided)

Output:
  <output_dir>/dataset   (DatasetDict with `input_ids` + `prompt_len`)
  <output_dir>/tokenizer

The pseudo-SFT data is intended for pipeline debugging and stability checks, not for
final instruction-following quality.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# This script is CPU-only; avoid accidental backend autoload issues.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import transformers
from datasets import DatasetDict, load_from_disk

from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_EOM, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_TOKEN
from dllm.utils.data import post_process_dataset
from dllm.utils.utils import get_default_logger


logger = get_default_logger(__name__)


@dataclass
class Args:
    pt_bundle: str = ""
    tokenizer_name_or_path: str = ""
    output_dir: str = "/tmp/oneflow_sft_pseudo_from_pt"
    strategy: str = "fixed_ratio"  # fixed_ratio | deterministic_uniform
    prompt_ratio: float = 0.5
    prompt_min_tokens: int = 32
    prompt_max_tokens: int = 512  # <=0 means no explicit max cap
    min_total_tokens: int = 8
    ensure_bos: bool = True
    seed: int = 42
    truncation: str = "right"  # right|filter
    max_length: int = 1024
    num_proc: int = 8


def build_tokenizer(name_or_path: str) -> transformers.PreTrainedTokenizer:
    tok = transformers.AutoTokenizer.from_pretrained(name_or_path, padding_side="right")
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


def _choose_prompt_len(
    total_len: int,
    *,
    strategy: str,
    prompt_ratio: float,
    prompt_min_tokens: int,
    prompt_max_tokens: int,
    sample_index: int,
    seed: int,
) -> int:
    # Leave at least one token for response.
    upper_hard = max(1, total_len - 1)
    lower = max(1, min(prompt_min_tokens, upper_hard))
    upper = upper_hard
    if int(prompt_max_tokens) > 0:
        upper = min(upper, int(prompt_max_tokens))
    upper = max(lower, upper)

    if strategy == "fixed_ratio":
        center = int(round(float(prompt_ratio) * float(total_len)))
        return int(max(lower, min(upper, center)))

    if strategy == "deterministic_uniform":
        span = int(upper - lower + 1)
        # Deterministic pseudo-random integer in [lower, upper], stable across runs.
        h = (int(sample_index) * 1103515245 + int(seed) * 12345 + 1013904223) & 0x7FFFFFFF
        return int(lower + (h % max(1, span)))

    raise ValueError(f"Unknown strategy: {strategy}. Expected fixed_ratio|deterministic_uniform")


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.pt_bundle:
        raise ValueError("--pt_bundle is required")

    pt_bundle = os.path.expanduser(str(args.pt_bundle))
    pt_dataset_dir = os.path.join(pt_bundle, "dataset")
    if not os.path.isdir(pt_dataset_dir):
        raise FileNotFoundError(f"Missing PT dataset dir: {pt_dataset_dir}")

    tok_path = args.tokenizer_name_or_path or os.path.join(pt_bundle, "tokenizer")
    tok_path = os.path.expanduser(str(tok_path))
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Tokenizer path not found: {tok_path}. "
            "Pass --tokenizer_name_or_path if PT bundle does not include tokenizer/."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading PT dataset from: {pt_dataset_dir}")
    ds = load_from_disk(pt_dataset_dir)
    if not isinstance(ds, DatasetDict):
        raise ValueError(f"Expected DatasetDict in {pt_dataset_dir}, got: {type(ds)}")
    if "train" not in ds:
        raise ValueError(f"Dataset has no 'train' split: splits={list(ds.keys())}")
    if "input_ids" not in ds["train"].column_names:
        raise ValueError("PT dataset split 'train' has no column 'input_ids'")

    tok = build_tokenizer(tok_path)
    bos_id = tok.bos_token_id

    logger.info(
        "Building pseudo-SFT with strategy=%s prompt_ratio=%.3f prompt_min=%d prompt_max=%d",
        args.strategy,
        float(args.prompt_ratio),
        int(args.prompt_min_tokens),
        int(args.prompt_max_tokens),
    )

    out_splits = {}
    for split_name, split_ds in ds.items():
        logger.info(f"Converting split: {split_name}, rows={len(split_ds)}")

        def _map_row(row, idx):
            ids = [int(x) for x in row["input_ids"]]
            if bool(args.ensure_bos) and (bos_id is not None):
                if not ids:
                    ids = [int(bos_id)]
                elif int(ids[0]) != int(bos_id):
                    ids = [int(bos_id)] + ids

            if len(ids) < int(args.min_total_tokens):
                return {"input_ids": ids, "prompt_len": 0, "__keep__": False}

            prompt_len = _choose_prompt_len(
                len(ids),
                strategy=str(args.strategy),
                prompt_ratio=float(args.prompt_ratio),
                prompt_min_tokens=int(args.prompt_min_tokens),
                prompt_max_tokens=int(args.prompt_max_tokens),
                sample_index=int(idx),
                seed=int(args.seed),
            )
            return {"input_ids": ids, "prompt_len": int(prompt_len), "__keep__": True}

        mapped = split_ds.map(
            _map_row,
            with_indices=True,
            remove_columns=split_ds.column_names,
            num_proc=int(args.num_proc),
            desc=f"Build pseudo-SFT fields ({split_name})",
        )
        mapped = mapped.filter(
            lambda row: bool(row["__keep__"]),
            num_proc=int(args.num_proc),
            desc=f"Filter short rows ({split_name})",
        )
        mapped = mapped.remove_columns(["__keep__"])
        out_splits[split_name] = mapped

    out = DatasetDict(out_splits)

    # Optional max_length post-process so downstream SFT training is predictable.
    class _PP:
        truncation = str(args.truncation)
        max_length = int(args.max_length)
        num_proc = int(args.num_proc)

    out = post_process_dataset(out, _PP)  # type: ignore[arg-type]

    out_ds_dir = os.path.join(args.output_dir, "dataset")
    out_tok_dir = os.path.join(args.output_dir, "tokenizer")
    logger.info(f"Saving pseudo-SFT dataset to: {out_ds_dir}")
    out.save_to_disk(out_ds_dir)
    logger.info(f"Saving tokenizer to: {out_tok_dir}")
    tok.save_pretrained(out_tok_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
