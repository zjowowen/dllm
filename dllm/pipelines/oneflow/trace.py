from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() <= 10_000:
            return x.detach().cpu().tolist()
        # Avoid huge blobs by default; keep shape only.
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return x


@dataclass
class OneFlowTrace:
    """
    A lightweight, JSON-serializable trace container for debugging OneFlow stages.

    This is intentionally permissive: you can attach any extra fields in `extra`.
    """

    seed: int | None = None
    tau_text: Any | None = None
    t_text: Any | None = None
    x1_ids: Any | None = None
    prompt_len: Any | None = None
    xt_ids: Any | None = None
    bags: Any | None = None
    images: Any | None = None
    unified: Any | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "seed": self.seed,
            "tau_text": _to_jsonable(self.tau_text),
            "t_text": _to_jsonable(self.t_text),
            "x1_ids": _to_jsonable(self.x1_ids),
            "prompt_len": _to_jsonable(self.prompt_len),
            "xt_ids": _to_jsonable(self.xt_ids),
            "bags": _to_jsonable(self.bags),
            "images": _to_jsonable(self.images),
            "unified": _to_jsonable(self.unified),
        }
        if self.extra:
            out["extra"] = _to_jsonable(self.extra)
        # drop Nones for cleanliness
        return {k: v for k, v in out.items() if v is not None}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=int(indent))


def format_unified_position_table(
    *,
    input_ids: list[int],
    is_any_modality: list[bool],
    times: list[float],
    image_token_id: int | None = None,
    pad_id: int | None = None,
    max_rows: int = 512,
) -> str:
    """
    Render a small markdown table for a single sample's unified sequence.

    This is meant for human debugging and docs; it intentionally avoids any tokenizer dependency.
    """
    rows = []
    n = min(len(input_ids), len(is_any_modality), len(times), int(max_rows))
    rows.append("| total_pos | kind | token_id | time | notes |")
    rows.append("|---:|---|---:|---:|---|")
    for i in range(n):
        tid = int(input_ids[i])
        is_mod = bool(is_any_modality[i])
        t = float(times[i])
        kind = "mod" if is_mod else "text"
        notes = ""
        if image_token_id is not None and (not is_mod) and tid == int(image_token_id):
            notes = "image_anchor"
        if pad_id is not None and tid == int(pad_id):
            notes = (notes + "," if notes else "") + "pad_id"
        rows.append(f"| {i} | {kind} | {tid} | {t:.4f} | {notes} |")
    if n < len(input_ids):
        rows.append(f"\n... truncated: showed {n} / {len(input_ids)} positions ...\n")
    return "\n".join(rows)


