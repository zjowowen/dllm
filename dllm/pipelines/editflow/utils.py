import math
import random
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Tuple

import torch
import transformers

from dllm.utils.utils import parse_spec
from dllm.pipelines.ctmc_utils import pad_1d


# ------------------------------- Collator (x0 source) --------------------------------
@dataclass
class X0Sampler:

    def __call__(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError("Subclasses must implement __call__.")


@dataclass
class SampleX0Empty(X0Sampler):
    """Return BOS-only (i.e., empty tail)."""

    tokenizer: transformers.PreTrainedTokenizer | None = None

    def __call__(self, *args, **kwargs) -> list[int]:
        return []


@dataclass
class SampleX0Masks(X0Sampler):
    """Return a run of mask tokens of given length."""

    length: int = 128
    tokenizer: transformers.PreTrainedTokenizer = None

    def __call__(self, *args, **kwargs) -> list[int]:
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("tokenizer needs mask_token_id for mask-based sampler")
        return [int(mask_id)] * self.length


# ---------------- Factory ---------------- #
_X0_SAMPLER_CLASSES: dict[str, type[X0Sampler]] = {
    "empty": SampleX0Empty,
    "masks": SampleX0Masks,
}


def make_x0_sampler(name: str, tokenizer: Any, **kwargs) -> X0Sampler:
    try:
        name, kvs = parse_spec(name)
        cls = _X0_SAMPLER_CLASSES[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown x0 sampler '{name}'. Available: {list(_X0_SAMPLER_CLASSES)}"
        )
    # merged_kwargs = {**kvs, **kwargs}
    return cls(tokenizer=tokenizer, **kvs, **kwargs)


@dataclass
class EditFlowCollator:
    tokenizer: transformers.PreTrainedTokenizer = None
    x0_sampler: Callable | str | None = X0Sampler  # can be func OR name

    def __post_init__(self):
        if isinstance(self.x0_sampler, str):
            self.x0_sampler = make_x0_sampler(self.x0_sampler, self.tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, list[Any]]:
        if not features:
            return {}

        keys = features[0].keys()
        batch = {k: [ex[k] for ex in features] for k in keys}
        batch["x1_ids"] = batch["input_ids"]

        if "prompt_len" not in batch:
            assert self.tokenizer.bos_token_id is not None
            bos = self.tokenizer.bos_token_id
            batch["x1_ids"] = [
                x if x and x[0] == bos else [bos] + x for x in batch["x1_ids"]
            ]
            batch["x0_ids"] = [
                x1_ids[:1] + self.x0_sampler(x1_ids=x1_ids[1:])
                for x1_ids in batch["x1_ids"]
            ]
        else:
            batch["x0_ids"] = [
                x1_ids[:prompt_len] + self.x0_sampler(x1_ids=x1_ids[prompt_len:])
                for x1_ids, prompt_len in zip(batch["x1_ids"], batch["prompt_len"])
            ]

        batch["return_loss"] = True
        return batch


def init_editflow_from_src(
    ef_model, src_model, lm_head_key: str = "lm_head", verbose: bool = True
):
    """
    Initialize an EditFlowModel (ef_model) from a pretrained source model.

    If DeepSpeed ZeRO-3 is enabled (detected via HF's `is_deepspeed_zero3_enabled()`),
    this function temporarily gathers full parameters for both models on rank 0,
    performs the copy there, and then returns to sharded mode automatically.
    Otherwise it behaves like a normal CPU/GPU single-process copy.

    Returns (missing_keys, unexpected_keys) from load_state_dict(strict=False).
    """
    import deepspeed
    from transformers.integrations import is_deepspeed_zero3_enabled

    dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if dist_ok else 0

    def _copy_once():
        src_sd = src_model.state_dict()
        tgt_sd = ef_model.state_dict()
        new_sd = OrderedDict()

        # 1) copy matching backbone tensors
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                new_sd[k] = v

        # 2) duplicate lm_head -> sub_logits & ins_logits (weight + optional bias)
        lm_w = f"{lm_head_key}.weight"
        lm_b = f"{lm_head_key}.bias"

        if lm_w in src_sd:
            if "sub_logits.weight" in tgt_sd:
                new_sd["sub_logits.weight"] = src_sd[lm_w]
            if "ins_logits.weight" in tgt_sd:
                new_sd["ins_logits.weight"] = src_sd[lm_w]
        if lm_b in src_sd:
            if "sub_logits.bias" in tgt_sd:
                new_sd["sub_logits.bias"] = src_sd[lm_b]
            if "ins_logits.bias" in tgt_sd:
                new_sd["ins_logits.bias"] = src_sd[lm_b]

        # 3) non-strict load so new rate heads remain randomly initialized
        missing, unexpected = ef_model.load_state_dict(new_sd, strict=False)
        return new_sd, missing, unexpected

    if is_deepspeed_zero3_enabled():
        # All ranks enter/exit together; only rank 0 materializes full tensors.
        params = list(ef_model.parameters()) + list(src_model.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if rank == 0:
                new_sd, missing, unexpected = _copy_once()
            else:
                new_sd, missing, unexpected = OrderedDict(), [], []

        if dist_ok:
            torch.distributed.barrier()

        if verbose and rank == 0:
            _p = getattr(globals().get("dllm", None), "utils", None)
            printer = getattr(_p, "print_main", print) if _p else print
            printer(
                f"[EditFlow init][ZeRO-3] Copied {len(new_sd)} tensors from Src Model."
            )
            if missing:
                printer("  Missing (expected for new rate heads, etc.):")
                for k in missing:
                    printer("   -", k)
            if unexpected:
                printer("  Unexpected (check key names):")
                for k in unexpected:
                    printer("   -", k)
        return missing, unexpected

    # --- Non-ZeRO (or DS not present) path ---
    new_sd, missing, unexpected = _copy_once()
    if verbose:
        _p = getattr(globals().get("dllm", None), "utils", None)
        printer = getattr(_p, "print_main", print) if _p else print
        printer(f"[EditFlow init] Copied {len(new_sd)} tensors from Src Model.")
        if missing:
            printer("  Missing (expected for new rate heads, etc.):")
            for k in missing:
                printer("   -", k)
        if unexpected:
            printer("  Unexpected (check key names):")
            for k in unexpected:
                printer("   -", k)
    return missing, unexpected


if __name__ == "__main__":
    pass
