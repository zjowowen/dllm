"""
Overfit inference check for OneFlow text-only checkpoints.

Goal:
  - Provide a *repeatable* sanity check that a text-only model can "memorize" a tiny
    training split, by sampling from a prefix and measuring overlap with the held-out
    continuation from the same training sample.

Notes:
  - OneFlow sampling is insertion-based (not autoregressive next-token), so perfect
    verbatim continuation is not guaranteed even when the training loss is low.
  - This script is designed to make the evaluation explicit and quantifiable.

Example (NPU):
  source activate_python_env.sh
  export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

  python -u scripts/oneflow/overfit_infer_check.py \
    --model_dir data/ckpts/oneflow_text_only_overfit256/checkpoint-final \
    --dataset_dir data/offline/pt_text_dclm_1024_overfit256/dataset \
    --tokenizer_dir data/offline/pt_text_dclm_1024_overfit256/tokenizer \
    --device npu \
    --sample_index 0 \
    --prefix_tokens 200 \
    --ref_tokens 200 \
    --temperature 0.9 \
    --dt 0.05 \
    --max_steps 512 \
    --max_new_tokens 256 \
    --max_seq_len 512 \
    --max_insertions_per_step 64 \
    --max_w 20 \
    --append_only False \
    --suppress_whitespace_tokens False \
    --output_dir data/vis/overfit_infer_check
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
import transformers


def _ngrams(seq: list[int], n: int) -> list[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(seq) < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def _multiset_f1(a: list[int], b: list[int]) -> dict[str, float]:
    """
    Multiset (Counter) overlap F1 on token ids.
    """
    ca = Counter(a)
    cb = Counter(b)
    overlap = 0
    for k, va in ca.items():
        vb = cb.get(k, 0)
        overlap += min(int(va), int(vb))
    prec = overlap / max(1, len(a))
    rec = overlap / max(1, len(b))
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _jaccard(a: Iterable[tuple[int, ...]], b: Iterable[tuple[int, ...]]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def _resolve_device(name: str) -> torch.device:
    name = str(name or "auto").lower()

    def _npu_available() -> bool:
        return bool(
            hasattr(torch, "npu")
            and hasattr(torch.npu, "is_available")
            and torch.npu.is_available()
        )

    if name == "auto":
        if _npu_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name in ("cpu", "cuda", "npu"):
        return torch.device(name)
    raise ValueError(f"Unknown --device: {name} (expected auto|cpu|cuda|npu)")


@dataclass
class Args:
    model_dir: str = ""
    dataset_dir: str = ""  # save_to_disk directory (DatasetDict or Dataset)
    tokenizer_dir: str | None = None  # if None, use model_dir
    device: str = "auto"

    sample_index: int = 0
    prefix_tokens: int = 200
    ref_tokens: int = 200

    # sampling params
    seed: int = 42
    dt: float = 0.05
    max_steps: int = 512
    temperature: float = 0.9
    use_pi_gate: bool = True
    append_only: bool = False
    suppress_whitespace_tokens: bool = False
    max_new_tokens: int | None = 256
    max_seq_len: int | None = 512
    max_insertions_per_step: int | None = 64
    max_w: float | None = 20.0

    output_dir: str = "data/vis/overfit_infer_check"


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required")

    device = _resolve_device(args.device)

    # Lazy imports for heavy deps.
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import `datasets`. If you are on a CPU-only machine with torch_npu installed, "
            "try running with environment variable: TORCH_DEVICE_BACKEND_AUTOLOAD=0"
        ) from e

    from dllm.pipelines.oneflow.models import OneFlowModel
    from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput

    os.makedirs(args.output_dir, exist_ok=True)

    tok_dir = args.tokenizer_dir or args.model_dir
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_dir)

    ds = load_from_disk(args.dataset_dir)
    # accept DatasetDict or Dataset
    if hasattr(ds, "keys"):
        if "train" not in ds:
            raise ValueError(f"dataset_dir has no 'train' split: splits={list(ds.keys())}")
        train = ds["train"]
    else:
        train = ds

    if "input_ids" not in train.column_names:
        raise ValueError("dataset must contain 'input_ids'")

    idx = int(args.sample_index)
    row = train[idx]
    ids = [int(x) for x in row["input_ids"]]
    prefix_n = max(1, int(args.prefix_tokens))
    ref_n = max(1, int(args.ref_tokens))

    prefix_ids = ids[:prefix_n]
    ref_ids = ids[prefix_n : prefix_n + ref_n]

    prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

    prompt_path = os.path.join(args.output_dir, "prompt.txt")
    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(ref_text)

    model = OneFlowModel.from_pretrained(args.model_dir, map_location="cpu").eval().to(device)
    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)

    cfg = OneFlowSamplerConfig(
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        temperature=float(args.temperature),
        use_pi_gate=bool(args.use_pi_gate),
        append_only=bool(args.append_only),
        suppress_whitespace_tokens=bool(args.suppress_whitespace_tokens),
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is None else int(args.max_new_tokens),
        max_seq_len=args.max_seq_len if args.max_seq_len is None else int(args.max_seq_len),
        max_insertions_per_step=args.max_insertions_per_step
        if args.max_insertions_per_step is None
        else int(args.max_insertions_per_step),
        max_w=args.max_w if args.max_w is None else float(args.max_w),
        image_num_tokens=0,  # text-only
        return_dict=True,
    )

    transformers.set_seed(int(args.seed))
    out = sampler.sample([prefix_ids], cfg, return_dict=True)
    assert isinstance(out, OneFlowSamplerOutput)
    out_ids = [int(x) for x in out.sequences[0].detach().to("cpu").tolist()]

    # Align by common prefix length (prompt is not edited by default).
    lcp = 0
    for a, b in zip(out_ids, prefix_ids):
        if int(a) != int(b):
            break
        lcp += 1

    gen_ids = out_ids[lcp:]
    # Compare only up to ref length.
    cmp_len = min(len(gen_ids), len(ref_ids))
    exact_matches = sum(1 for i in range(cmp_len) if int(gen_ids[i]) == int(ref_ids[i]))
    exact_rate = exact_matches / max(1, cmp_len)

    uni_f1 = _multiset_f1(gen_ids[:ref_n], ref_ids)
    bi_j = _jaccard(_ngrams(gen_ids[:ref_n], 2), _ngrams(ref_ids, 2))
    four_j = _jaccard(_ngrams(gen_ids[:ref_n], 4), _ngrams(ref_ids, 4))

    gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    gen_path = os.path.join(args.output_dir, "gen.txt")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(gen_text)

    report = {
        "model_dir": str(args.model_dir),
        "dataset_dir": str(args.dataset_dir),
        "tokenizer_dir": str(tok_dir),
        "device": str(device),
        "sample_index": int(idx),
        "prefix_tokens": int(prefix_n),
        "ref_tokens": int(ref_n),
        "prompt_prefix_match": bool(lcp == len(prefix_ids)),
        "prompt_lcp_tokens": int(lcp),
        "gen_tokens_total": int(len(out_ids)),
        "gen_tokens_after_prompt": int(len(gen_ids)),
        "ref_tokens_total": int(len(ref_ids)),
        "exact_match_rate_aligned": float(exact_rate),
        "unigram_multiset_f1": uni_f1,
        "bigram_jaccard": float(bi_j),
        "4gram_jaccard": float(four_j),
        "paths": {"prompt": prompt_path, "ref": ref_path, "gen": gen_path},
        "sampling": {
            "seed": int(args.seed),
            "dt": float(args.dt),
            "max_steps": int(args.max_steps),
            "temperature": float(args.temperature),
            "use_pi_gate": bool(args.use_pi_gate),
            "append_only": bool(args.append_only),
            "suppress_whitespace_tokens": bool(args.suppress_whitespace_tokens),
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "max_insertions_per_step": args.max_insertions_per_step,
            "max_w": args.max_w,
        },
    }

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Overfit inference check ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSaved:")
    print(" -", prompt_path)
    print(" -", ref_path)
    print(" -", gen_path)
    print(" -", report_path)


if __name__ == "__main__":
    main()

