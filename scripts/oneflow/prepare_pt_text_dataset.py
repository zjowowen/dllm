"""
Prepare an offline, pre-tokenized PT dataset (contains `input_ids`) and save with
🤗 datasets `save_to_disk`, so it can be trained on a no-network cluster via
`examples/oneflow/pt_text.py --load_preprocessed_data True --dataset_args <path>`.

This script is meant to run on a machine WITH internet access (or where the raw
dataset is accessible), and then you copy the output folder to the cluster.

Example:
  python -u scripts/oneflow/prepare_pt_text_dataset.py \
    --dataset_name_or_path Trelis/tiny-shakespeare \
    --text_field Text \
    --tokenizer_name_or_path gpt2 \
    --seq_length 256 \
    --output_dir /tmp/oneflow_pt_tinyshakespeare_256
"""

import functools
import os
from dataclasses import dataclass

import transformers
from datasets import DatasetDict, load_dataset

from dllm.pipelines.oneflow.utils import ONEFLOW_IMAGE_EOM, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_TOKEN
from dllm.utils.utils import get_default_logger
from dllm.utils import tokenize_and_group

logger = get_default_logger(__name__)


@dataclass
class Args:
    dataset_name_or_path: str = "Trelis/tiny-shakespeare"
    dataset_config_name: str | None = None
    train_split: str = "train"
    test_split: str | None = "test"
    text_field: str = "Text"

    tokenizer_name_or_path: str = "gpt2"
    seq_length: int = 256
    insert_eos: bool = True
    drop_tail: bool = True

    # optional limits for quick exports
    train_limit: int | None = None
    test_limit: int | None = None

    output_dir: str = "/tmp/oneflow_pt_text"
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


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Building tokenizer: {args.tokenizer_name_or_path}")
    tok = build_tokenizer(args.tokenizer_name_or_path)

    logger.info(f"Loading raw dataset: {args.dataset_name_or_path}")
    raw = load_dataset(args.dataset_name_or_path, name=args.dataset_config_name)
    ds = DatasetDict()
    ds["train"] = raw[args.train_split]
    if args.test_split:
        ds["test"] = raw[args.test_split]

    if args.train_limit is not None:
        ds["train"] = ds["train"].select(range(min(args.train_limit, len(ds["train"]))))
    if args.test_split and args.test_limit is not None:
        ds["test"] = ds["test"].select(range(min(args.test_limit, len(ds["test"]))))

    map_fn = functools.partial(
        tokenize_and_group,
        tokenizer=tok,
        text_field=args.text_field,
        seq_length=args.seq_length,
        insert_eos=args.insert_eos,
        drop_tail=args.drop_tail,
        add_special_tokens=False,
    )

    logger.info("Tokenizing & grouping...")
    out = ds.map(
        map_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=args.num_proc,
        desc="Mapping dataset to PT format",
    )

    # Ensure BOS
    bos_id = int(tok.bos_token_id)

    def add_bos(row):
        ids = row["input_ids"]
        if ids and ids[0] != bos_id:
            row["input_ids"] = [bos_id] + ids
        return row

    out = out.map(add_bos, num_proc=args.num_proc, desc="Prepending BOS")

    # Save tokenizer alongside dataset for offline use
    tok_dir = os.path.join(args.output_dir, "tokenizer")
    tok.save_pretrained(tok_dir)

    ds_dir = os.path.join(args.output_dir, "dataset")
    logger.info(f"Saving dataset to: {ds_dir}")
    out.save_to_disk(ds_dir)

    logger.info("Done.")
    logger.info(f"Tokenizer: {tok_dir}")
    logger.info(f"Dataset:   {ds_dir}")


if __name__ == "__main__":
    main()


