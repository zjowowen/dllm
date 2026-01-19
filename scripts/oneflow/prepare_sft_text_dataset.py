"""
Prepare an offline, pre-tokenized SFT dataset (contains `input_ids` and optional `prompt_len`)
and save with 🤗 datasets `save_to_disk`.

This is meant to run on a machine WITH internet access (or where the raw dataset is accessible),
and then you copy the output folder to the cluster.

Output layout:
  <output_dir>/dataset   (DatasetDict saved to disk)
  <output_dir>/tokenizer (tokenizer files saved to disk)

Example:
  python -u scripts/oneflow/prepare_sft_text_dataset.py \
    --dataset_args "HuggingFaceH4/ultrachat_200k[train:50000]" \
    --tokenizer_name_or_path gpt2 \
    --max_length 1024 \
    --output_dir /tmp/oneflow_sft_ultrachat_50k_1024
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Avoid failing on non-Ascend machines that have torch_npu installed but do not have
# Ascend runtime libraries (e.g. libhccl.so) configured. This script is CPU-only.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import transformers
from datasets import DatasetDict

import dllm
from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_EOM, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_TOKEN
from dllm.utils.data import post_process_dataset
from dllm.utils.utils import get_default_logger


logger = get_default_logger(__name__)


@dataclass
class Args:
    dataset_args: str = "tatsu-lab/alpaca"
    tokenizer_name_or_path: str = "gpt2"
    max_length: int = 1024
    truncation: str = "right"  # right|filter
    mask_prompt_loss: bool = True
    num_proc: int = 8
    output_dir: str = "/tmp/oneflow_sft_text"


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


def sft_map_fn(row, *, tokenizer: transformers.PreTrainedTokenizer, mask_prompt_loss: bool) -> dict:
    """
    Convert a chat-style sample (row['messages']) into OneFlow SFT fields.

    - Prefer tokenizer.apply_chat_template when available.
    - Otherwise fallback to a plain-text format: prompt + "\\n\\n" + response.
    """
    messages = row["messages"]

    def ensure_bos(ids: list[int]) -> list[int]:
        bos = tokenizer.bos_token_id
        if bos is None:
            return [int(x) for x in ids]
        if not ids:
            return [int(bos)]
        if int(ids[0]) != int(bos):
            return [int(bos)] + [int(x) for x in ids]
        return [int(x) for x in ids]

    use_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        full_ids = ensure_bos(full_ids)
        if mask_prompt_loss:
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
            prompt_ids = ensure_bos(prompt_ids)
            return {"input_ids": full_ids, "prompt_len": len(prompt_ids)}
        return {"input_ids": full_ids}

    # Plain-text fallback (works for GPT-like tokenizers).
    if not messages or len(messages) < 2:
        raise ValueError("Expected at least 2 messages (user + assistant).")
    prompt_text = str(messages[0].get("content", "") or "").strip()
    resp_text = str(messages[-1].get("content", "") or "").strip()
    prompt_with_delim = prompt_text + "\n\n"
    full_text = prompt_with_delim + resp_text

    full_ids = ensure_bos(tokenizer.encode(full_text, add_special_tokens=False))
    if mask_prompt_loss:
        prompt_ids = ensure_bos(tokenizer.encode(prompt_with_delim, add_special_tokens=False))
        return {"input_ids": full_ids, "prompt_len": len(prompt_ids)}
    return {"input_ids": full_ids}


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Building tokenizer: {args.tokenizer_name_or_path}")
    tok = build_tokenizer(args.tokenizer_name_or_path)

    logger.info(f"Loading SFT dataset: {args.dataset_args}")
    with transformers.utils.logging.disable_progress_bar():
        ds: DatasetDict = dllm.data.load_sft_dataset(args.dataset_args, load_preprocessed_data=False)

    # Map to {input_ids, prompt_len?}
    ds = ds.map(
        lambda row: sft_map_fn(row, tokenizer=tok, mask_prompt_loss=bool(args.mask_prompt_loss)),
        remove_columns=ds["train"].column_names,
        num_proc=int(args.num_proc),
        desc="Mapping dataset to SFT token ids",
    )

    # Post-process to max_length (filter or right-truncate).
    class _PP:
        truncation = str(args.truncation)
        max_length = int(args.max_length)
        num_proc = int(args.num_proc)

    ds = post_process_dataset(ds, _PP)  # type: ignore[arg-type]

    out_ds_dir = os.path.join(args.output_dir, "dataset")
    out_tok_dir = os.path.join(args.output_dir, "tokenizer")

    logger.info(f"Saving dataset to: {out_ds_dir}")
    ds.save_to_disk(out_ds_dir)
    logger.info(f"Saving tokenizer to: {out_tok_dir}")
    tok.save_pretrained(out_tok_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()

