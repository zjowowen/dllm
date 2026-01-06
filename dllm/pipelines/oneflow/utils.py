from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


ONEFLOW_IMAGE_TOKEN = "<|oneflow_image|>"
ONEFLOW_IMAGE_SOM = "<|oneflow_image_som|>"
ONEFLOW_IMAGE_EOM = "<|oneflow_image_eom|>"


@dataclass
class OneFlowCollator:
    """
    Data collator for OneFlow.

    Expected per-sample input (recommended v1):
    - { "segments": list[Tensor], ... } where each element is:
        - torch.long [seq]  (text tokens, may contain `<|oneflow_image|>`)
        - torch.float [...] (image latent tensor aligned to the preceding image token)
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # NOTE: we intentionally keep `x1_ids` as python lists (variable length),
        # matching the style used by `dllm/pipelines/editflow`.
        x1_ids = [f["input_ids"] for f in features]
        out: Dict[str, Any] = {"x1_ids": x1_ids}
        if "prompt_len" in features[0]:
            out["prompt_len"] = [int(f.get("prompt_len", 0)) for f in features]
        # Optional: pass through image latents (pre-encoded) for mixed-modal training.
        # Convention: each feature may have:
        # - "image_latents": Tensor or list[Tensor]
        # - or "image_latent": Tensor
        if "image_latents" in features[0]:
            out["image_latents"] = [f.get("image_latents") for f in features]
        elif "image_latent" in features[0]:
            out["image_latents"] = [f.get("image_latent") for f in features]
        return out


def insert_token(x: torch.Tensor, idx: int, token_id: int) -> torch.Tensor:
    """
    Insert `token_id` into 1D tensor `x` at position idx (python slicing semantics).
    """
    if x.ndim != 1:
        raise ValueError(f"expected 1D tensor, got shape {tuple(x.shape)}")
    idx = int(max(0, min(idx, x.numel())))
    token = torch.tensor([int(token_id)], dtype=x.dtype, device=x.device)
    return torch.cat([x[:idx], token, x[idx:]], dim=0)


