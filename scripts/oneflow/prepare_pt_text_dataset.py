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
import logging
import os
import random
import time
from dataclasses import dataclass

# Avoid failing on non-Ascend machines that have torch_npu installed but do not have
# Ascend runtime libraries (e.g. libhccl.so) configured. This script is CPU-only.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import transformers
from datasets import Dataset, DatasetDict

from dllm.data.utils import _load_dataset_with_retry

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
    # `tokenize_and_group` returns both input_ids and labels (labels==input_ids).
    # OneFlow text-only pretraining only needs input_ids, so we drop labels by default
    # to reduce disk usage for large corpora.
    keep_labels: bool = False

    # Use HF streaming mode to avoid downloading the full corpus.
    # Recommended for very large datasets (e.g., dclm-baseline) when you only need
    # a limited subset for offline export.
    streaming: bool = False
    tokenizer_batch_size: int = 256

    # optional limits for quick exports
    train_limit: int | None = None
    test_limit: int | None = None

    output_dir: str = "/tmp/oneflow_pt_text"
    num_proc: int = 8

    # Streaming retry policy (only applies when --streaming True).
    # We intentionally "retry from scratch" because robust continuation on IterableDataset
    # is hard to guarantee (may cause duplicates/missing rows).
    streaming_max_retries: int = 8
    streaming_retry_backoff_base_s: float = 2.0
    streaming_retry_backoff_max_s: float = 60.0
    streaming_retry_jitter_ratio: float = 0.2

    # Write a persistent log for postmortem debugging.
    # If empty, defaults to <output_dir>/prepare_pt_text_dataset.log
    log_file: str | None = None


def _setup_file_logging(*, logger: logging.Logger, log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Avoid adding duplicate file handlers if main() is called multiple times.
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == os.path.abspath(log_path):
            return
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)


def _is_transient_streaming_error(e: BaseException) -> bool:
    # We keep this simple and string-based to avoid tight coupling to optional deps.
    msg = str(e).lower()
    transient_markers = (
        "incompleteread",
        "chunkedencodingerror",
        "connection broken",
        "protocolerror",
        "read timed out",
        "readtimeout",
        "connection reset",
        "connection aborted",
        "remote end closed connection",
        "temporary failure",
        "timed out",
        "502",
        "503",
        "504",
        "429",
    )
    return any(m in msg for m in transient_markers)


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
    log_path = (
        str(args.log_file).strip()
        if args.log_file is not None and str(args.log_file).strip()
        else os.path.join(args.output_dir, "prepare_pt_text_dataset.log")
    )
    _setup_file_logging(logger=logger, log_path=log_path)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Building tokenizer: {args.tokenizer_name_or_path}")
    tok = build_tokenizer(args.tokenizer_name_or_path)

    # Allow disabling test split via CLI: --test_split None / "" / null
    if args.test_split is not None and str(args.test_split).strip().lower() in ("", "none", "null", "nil"):
        args.test_split = None

    logger.info(f"Loading raw dataset: {args.dataset_name_or_path} (streaming={bool(args.streaming)})")
    raw = _load_dataset_with_retry(
        args.dataset_name_or_path,
        name=args.dataset_config_name,
        streaming=bool(args.streaming),
    )

    def build_pt_split_from_streaming(
        split_iter,
        *,
        text_field: str,
        limit: int | None,
    ) -> Dataset:
        """
        Stream rows -> tokenize -> concatenate -> chunk into fixed-length input_ids.

        Note:
        - `limit` is number of raw rows consumed from the streaming dataset.
        - resulting number of sequences depends on total token count.
        """
        it = split_iter.take(int(limit)) if limit is not None else split_iter

        eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else None

        buf: list[int] = []
        buf_pos = 0
        seqs: list[list[int]] = []

        def flush_text_batch(texts: list[str]) -> None:
            nonlocal buf, buf_pos
            if not texts:
                return
            out_tok = tok(texts, add_special_tokens=False)
            for ids in out_tok["input_ids"]:
                if bool(args.insert_eos) and eos_id is not None:
                    if (not ids) or int(ids[-1]) != eos_id:
                        ids = list(ids) + [eos_id]
                buf.extend([int(x) for x in ids])
                while (len(buf) - buf_pos) >= int(args.seq_length):
                    start = buf_pos
                    end = buf_pos + int(args.seq_length)
                    seqs.append(buf[start:end])
                    buf_pos = end
                # compact occasionally to avoid unbounded growth
                if buf_pos > 1_000_000:
                    buf = buf[buf_pos:]
                    buf_pos = 0

        batch: list[str] = []
        n_rows = 0
        bs = max(1, int(args.tokenizer_batch_size))
        try:
            for row in it:
                if text_field not in row:
                    raise KeyError(
                        f"Missing text_field='{text_field}' in row keys={list(row.keys())}"
                    )
                t = row[text_field]
                if not isinstance(t, str):
                    t = str(t)
                batch.append(t)
                n_rows += 1
                if len(batch) >= bs:
                    flush_text_batch(batch)
                    batch = []
        except ValueError as e:
            # Common for corpora stored as .zst when zstd decoder is missing.
            if "compression type zstd not supported" in str(e).lower():
                raise RuntimeError(
                    "zstd (.zst) compression is not supported in this environment.\n"
                    "Fix: `pip install zstandard` (recommended) or `pip install pyzstd`, then rerun."
                ) from e
            raise

        if batch:
            flush_text_batch(batch)

        if not bool(args.drop_tail):
            rem = buf[buf_pos:]
            if rem:
                seqs.append(rem)

        logger.info(f"Streaming split consumed rows={n_rows}, produced sequences={len(seqs)}")
        if bool(args.keep_labels):
            return Dataset.from_dict({"input_ids": seqs, "labels": [s[:] for s in seqs]})
        return Dataset.from_dict({"input_ids": seqs})

    if bool(args.streaming):
        # Streaming path: make train_limit/test_limit actually limit network IO.
        ds_out = DatasetDict()

        def build_split_with_retries(split_name: str, split_iter, *, limit: int | None):
            max_retries = max(0, int(args.streaming_max_retries))
            base = float(args.streaming_retry_backoff_base_s)
            backoff_max = float(args.streaming_retry_backoff_max_s)
            jitter = float(args.streaming_retry_jitter_ratio)

            attempt = 0
            while True:
                try:
                    logger.info(
                        f"[streaming] Build split='{split_name}' attempt={attempt + 1}/{max_retries + 1} "
                        f"(limit={limit})"
                    )
                    return build_pt_split_from_streaming(
                        split_iter,
                        text_field=args.text_field,
                        limit=limit,
                    )
                except Exception as e:
                    # Non-transient errors should fail fast (schema error, zstd missing, etc).
                    if not _is_transient_streaming_error(e):
                        logger.exception(
                            f"[streaming] Non-transient error while building split='{split_name}'. "
                            f"Not retrying."
                        )
                        raise

                    if attempt >= max_retries:
                        logger.exception(
                            f"[streaming] Transient error while building split='{split_name}', "
                            f"retries exhausted (attempts={max_retries + 1})."
                        )
                        raise

                    # Exponential backoff with jitter.
                    wait = min(backoff_max, base * (2**attempt))
                    wait = wait * (1.0 + random.uniform(-jitter, jitter))
                    wait = max(0.0, float(wait))
                    logger.exception(
                        f"[streaming] Transient error while building split='{split_name}' "
                        f"(attempt={attempt + 1}/{max_retries + 1}). Retrying from scratch in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    attempt += 1

        ds_out["train"] = build_split_with_retries(
            "train",
            raw[args.train_split],
            limit=args.train_limit,
        )
        if args.test_split and args.test_split in raw:
            ds_out["test"] = build_split_with_retries(
                "test",
                raw[args.test_split],
                limit=args.test_limit,
            )
        out = ds_out
    else:
        # Non-streaming path: downloads/prepare full dataset (can be huge).
        ds = DatasetDict()
        ds["train"] = raw[args.train_split]
        if args.test_split:
            if args.test_split in raw:
                ds["test"] = raw[args.test_split]
            else:
                logger.warning(
                    f"Requested test_split='{args.test_split}' but dataset has splits={list(raw.keys())}. "
                    "Skipping test split."
                )

        if args.train_limit is not None:
            ds["train"] = ds["train"].select(range(min(args.train_limit, len(ds["train"]))))
        if "test" in ds and args.test_limit is not None:
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

    if (not bool(args.keep_labels)) and ("labels" in out["train"].column_names):
        # Drop labels to save disk (OneFlow collator only uses `input_ids`).
        out = DatasetDict(
            {
                name: (split.remove_columns("labels") if "labels" in split.column_names else split)
                for name, split in out.items()
            }
        )

    # Ensure BOS
    bos_id = int(tok.bos_token_id)

    def add_bos(row):
        ids = row["input_ids"]
        if ids and ids[0] != bos_id:
            row["input_ids"] = [bos_id] + ids
            if "labels" in row and isinstance(row["labels"], list):
                row["labels"] = [bos_id] + row["labels"]
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


