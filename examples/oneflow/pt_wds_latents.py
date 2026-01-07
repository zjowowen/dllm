"""
OneFlow multimodal pretraining from **offline** WebDataset latents shards.

This is the recommended training entrypoint for a no-network cluster:
- You precompute `latent.npy` (+ optional `input_ids.json`) shards with
  `scripts/oneflow/precompute_latents_wds.py` on a machine WITH internet/data access.
- Copy the produced bundle to the cluster.
- Train from local shard paths only (no HF downloads).

Expected WebDataset sample fields (per item in tar):
  - `npy`: latent array, shape [4, H, W] (SD-VAE scaled, e.g., 0.18215 * z)
  - `txt`: caption text (utf-8)
  - optional `json`: {"input_ids": [...]} where input_ids already includes
      [BOS] + caption_ids + [<|oneflow_image|>] + [EOS]

Example (offline, using precomputed input_ids):
  accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/oneflow/pt_wds_latents.py \
    --output_dir /tmp/oneflow_mm_latents \
    --tokenizer_name_or_path /path/to/bundle/tokenizer \
    --shards "/path/to/bundle/wds_latents/shard-{000000..000099}.tar" \
    --use_precomputed_ids True \
    --max_steps 2000
"""

from __future__ import annotations

import io
import json
import os
import glob
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, List

import accelerate
import numpy as np
import torch
import transformers
import webdataset as wds

import dllm
from dllm.pipelines.oneflow import OneFlowTrainer
from dllm.pipelines.oneflow.models import OneFlowConfig, OneFlowModel
from dllm.pipelines.oneflow.utils import (
    ONEFLOW_IMAGE_EOM,
    ONEFLOW_IMAGE_SOM,
    ONEFLOW_IMAGE_TOKEN,
    OneFlowCollator,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = None  # overwrite this (offline tokenizer dir)
    init_model_dir: str | None = None  # optional: init from a text-only checkpoint

    # model size (used only when init_model_dir is None)
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8

    # latent dim
    dim_latent: int = 4


@dataclass
class DataArguments:
    shards: str = None  # overwrite this (local tar shards, braceexpand ok)

    latent_key: str = "npy"
    caption_key: str = "txt"
    meta_key: str = "json"
    meta_input_ids_field: str = "input_ids"

    use_precomputed_ids: bool = True
    max_caption_tokens: int = 128  # only used when use_precomputed_ids=False

    # shuffling
    # WebDataset expects an int (e.g., 1000) or 0/False
    shardshuffle: int = 1000
    shuffle_buffer: int = 10_000

    # Stop after N samples (debug)
    max_samples: int | None = None


@dataclass
class TrainingArguments(OneFlowTrainer.OneFlowConfig):
    output_dir: str = None  # overwrite this
    num_train_epochs: float = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_strategy: str = "no"
    dataloader_num_workers: int = 4
    scheduler_cls: str = field(default="LinearKappaScheduler")


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
      - a directory containing *.tar
      - a text file listing shards (one per line)
    """
    spec = str(spec)
    if os.path.isdir(spec):
        # local directory
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

    # braceexpand (webdataset helper)
    if any(ch in spec for ch in ["*", "?", "[", "]"]):
        out = sorted(glob.glob(spec))
        if not out:
            raise FileNotFoundError(f"No shards matched glob: {spec}")
        return out

    out = list(wds.shardlists.expand_urls(spec))
    if not out:
        raise FileNotFoundError(f"No shards matched spec: {spec}")
    return out


class WDSLatentsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        shards: list[str],
        tokenizer: transformers.PreTrainedTokenizer,
        latent_key: str,
        caption_key: str,
        meta_key: str,
        meta_input_ids_field: str,
        use_precomputed_ids: bool,
        max_caption_tokens: int,
        shardshuffle: bool,
        shuffle_buffer: int,
        max_samples: int | None,
        seed: int,
    ):
        super().__init__()
        self._all_shards = shards
        self.tokenizer = tokenizer
        self.latent_key = str(latent_key)
        self.caption_key = str(caption_key)
        self.meta_key = str(meta_key)
        self.meta_input_ids_field = str(meta_input_ids_field)
        self.use_precomputed_ids = bool(use_precomputed_ids)
        self.max_caption_tokens = int(max_caption_tokens)
        self.shardshuffle = bool(shardshuffle)
        self.shuffle_buffer = int(shuffle_buffer)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.seed = int(seed)

        # Special token ids
        self.bos_id = int(tokenizer.bos_token_id)
        self.eos_id = int(tokenizer.eos_token_id)
        self.image_token_id = int(tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))
        if tokenizer.unk_token_id is not None and self.image_token_id == int(tokenizer.unk_token_id):
            raise ValueError(
                f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}. "
                "Please add it as a special token before training."
            )

        # Shard by global process rank (disjoint shards per process) for efficient IO
        st = accelerate.PartialState()
        self.rank = int(st.process_index)
        self.world = int(st.num_processes)
        self._my_shards = shards[self.rank::self.world] if self.world > 1 else shards
        if not self._my_shards:
            raise ValueError(
                f"Shard split produced empty shard list for rank={self.rank} world={self.world}. "
                f"Total shards={len(shards)}"
            )

    def _to_feature(self, sample: dict[str, Any]) -> dict[str, Any]:
        # latent (npy)
        lat = sample.get(self.latent_key, None)
        if lat is None:
            raise KeyError(f"Missing latent key '{self.latent_key}'")
        if isinstance(lat, (bytes, bytearray, memoryview)):
            lat = np.load(io.BytesIO(lat))  # expects .npy bytes
        lat_t = torch.from_numpy(np.asarray(lat))

        # input_ids
        if self.use_precomputed_ids:
            meta = sample.get(self.meta_key, None)
            if meta is None:
                raise KeyError(
                    f"use_precomputed_ids=True but missing meta key '{self.meta_key}'."
                )
            if isinstance(meta, (bytes, bytearray, memoryview)):
                meta = json.loads(meta)
            if not isinstance(meta, dict) or self.meta_input_ids_field not in meta:
                raise ValueError(
                    f"Invalid meta content; expected dict with '{self.meta_input_ids_field}'."
                )
            x1 = meta[self.meta_input_ids_field]
            if not isinstance(x1, list) or not x1:
                raise ValueError("Invalid input_ids in meta")
            input_ids = [int(t) for t in x1]
        else:
            cap = sample.get(self.caption_key, None)
            if cap is None:
                raise KeyError(f"Missing caption key '{self.caption_key}'")
            if isinstance(cap, (bytes, bytearray, memoryview)):
                cap = cap.decode("utf-8", errors="ignore")
            cap = str(cap).strip()
            if not cap:
                raise ValueError("Empty caption")
            ids = self.tokenizer.encode(cap, add_special_tokens=False)[: self.max_caption_tokens]
            input_ids = [self.bos_id] + [int(t) for t in ids] + [self.image_token_id] + [self.eos_id]

        return {"input_ids": input_ids, "image_latent": lat_t}

    def __iter__(self) -> Iterable[dict[str, Any]]:
        ds = wds.WebDataset(
            self._my_shards,
            shardshuffle=int(self.shardshuffle),
            handler=wds.warn_and_continue,
            workersplitter=wds.split_by_worker,
        )

        if self.shuffle_buffer > 0:
            wi = torch.utils.data.get_worker_info()
            wid = int(wi.id) if wi is not None else 0
            rng = random.Random(int(self.seed) + 10_000 * int(self.rank) + wid)
            ds = ds.shuffle(self.shuffle_buffer, initial=self.shuffle_buffer, rng=rng)

        ds = ds.map(self._to_feature, handler=wds.warn_and_continue)
        ds = ds.repeat()

        if self.max_samples is None:
            yield from ds
        else:
            # Manual take() for portability
            for i, item in enumerate(ds):
                if i >= self.max_samples:
                    break
                yield item


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # label-free training
    training_args.label_names = []
    training_args.remove_unused_columns = False
    # important for iterable datasets
    training_args.accelerator_config.dispatch_batches = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    if not model_args.tokenizer_name_or_path:
        raise ValueError("--tokenizer_name_or_path is required (local offline tokenizer dir).")
    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    if not data_args.shards:
        raise ValueError("--shards is required (local tar shard paths / pattern).")
    shards = expand_shards(data_args.shards)

    # dataset
    dataset = WDSLatentsIterableDataset(
        shards=shards,
        tokenizer=tokenizer,
        latent_key=data_args.latent_key,
        caption_key=data_args.caption_key,
        meta_key=data_args.meta_key,
        meta_input_ids_field=data_args.meta_input_ids_field,
        use_precomputed_ids=data_args.use_precomputed_ids,
        max_caption_tokens=data_args.max_caption_tokens,
        shardshuffle=data_args.shardshuffle,
        shuffle_buffer=data_args.shuffle_buffer,
        max_samples=data_args.max_samples,
        seed=training_args.seed,
    )

    # model
    if model_args.init_model_dir:
        logger.info(f"Loading OneFlowModel from: {model_args.init_model_dir}")
        model = OneFlowModel.from_pretrained(model_args.init_model_dir)
        # ensure tokenizer size matches (e.g., after adding special tokens)
        model.resize_token_embeddings(len(tokenizer))
    else:
        cfg = OneFlowConfig(
            vocab_size=len(tokenizer),
            bos_token_id=int(tokenizer.bos_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            unk_token_id=int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
            dim=model_args.dim,
            depth=model_args.depth,
            dim_head=model_args.dim_head,
            heads=model_args.heads,
            dim_latent=model_args.dim_latent,
        )
        model = OneFlowModel(cfg)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow multimodal training from offline latents shards...")
    trainer = OneFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
        data_collator=OneFlowCollator(tokenizer=tokenizer),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
    )
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()


