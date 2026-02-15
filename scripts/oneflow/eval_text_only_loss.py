"""
Offline evaluation for OneFlow text-only checkpoints using the paper Eq(7) text loss.

This is meant as a *sanity check* that training actually learned something, without
depending on the (non-autoregressive) insertion sampler quality.

It computes the same loss components as training:
  - token CE over bag-of-tokens targets
  - pi BCE
  - lambda_nonzero Poisson term (k>0)

Example (NPU):
  source activate_python_env.sh
  python -u scripts/oneflow/eval_text_only_loss.py \
    --model_dir data/ckpts/oneflow_text_only_pt_dclm20k/checkpoint-final \
    --dataset_dir data/offline/pt_text_dclm_1024_dbg/dataset \
    --device npu \
    --batch_size 8 \
    --num_batches 50 \
    --compare_random True
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import transformers
from datasets import load_from_disk

import dllm
from dllm.core.schedulers import make_kappa_scheduler
from dllm.pipelines.ctmc_utils import pad_1d
from dllm.pipelines.oneflow.losses import text_loss_paper_eq7_fast
from dllm.pipelines.oneflow.models import OneFlowConfig, OneFlowModel
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_EVAL_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)
from dllm.pipelines.oneflow.sequence_ops import build_noised_xt_and_bags, sample_tau_text, tau_to_t_text


@dataclass
class Args:
    model_dir: str = ""
    dataset_dir: str = ""
    tokenizer_dir: str | None = None
    device: str = "auto"  # auto|cpu|cuda|npu

    batch_size: int = 8
    num_batches: int = 50
    seed: int = 42
    seeds: str = ""  # Optional comma-separated list, e.g. "41,42,43"

    # Sampling/noising config (match training defaults)
    condition_text_on_time: bool | None = None
    scheduler_cls: str | None = None
    tau_text_max: float | None = None

    compare_random: bool = False
    output_json: str = ""


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


def _parse_seeds(seed: int, seeds_raw: str) -> list[int]:
    raw = str(seeds_raw or "").strip()
    if not raw:
        return [int(seed)]
    out: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        s = int(p)
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    if not out:
        raise ValueError("--seeds was provided but no valid integers were parsed.")
    return out


def _summarize_metrics(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not per_seed:
        raise ValueError("per_seed must contain at least one metrics row.")
    metric_keys = ("loss_total", "loss_tok", "loss_pi", "loss_lam")
    out: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        vals = [float(row[key]) for row in per_seed]
        n = len(vals)
        mean = sum(vals) / float(n)
        if n > 1:
            var = sum((x - mean) ** 2 for x in vals) / float(n - 1)
            std = math.sqrt(max(var, 0.0))
            sem = std / math.sqrt(float(n))
            ci95 = 1.96 * sem
        else:
            std = 0.0
            sem = 0.0
            ci95 = 0.0
        svals = sorted(vals)
        p50 = svals[n // 2] if n % 2 == 1 else 0.5 * (svals[n // 2 - 1] + svals[n // 2])
        p95_idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
        out[key] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci95": ci95,
            "min": svals[0],
            "p50": p50,
            "p95": svals[p95_idx],
            "max": svals[-1],
            "n": int(n),
        }
    return out


@torch.no_grad()
def _eval_model(
    *,
    model: OneFlowModel,
    tokenizer: transformers.PreTrainedTokenizer,
    train_ds,
    device: torch.device,
    scheduler_cls: str,
    batch_size: int,
    num_batches: int,
    seed: int,
    condition_text_on_time: bool,
    tau_text_max: float,
) -> dict[str, float]:
    model = model.to(device).eval()
    scheduler = make_kappa_scheduler(str(scheduler_cls))

    rng = random.Random(int(seed))
    n = int(len(train_ds))
    total_samples = int(batch_size) * int(num_batches)
    if total_samples <= 0:
        raise ValueError("batch_size*num_batches must be > 0")

    # Sample indices with replacement to avoid requiring n >= total_samples.
    idxs = [rng.randrange(n) for _ in range(total_samples)]

    pad_id = int(tokenizer.pad_token_id)
    bos_id = int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None

    sums = {"loss_total": 0.0, "loss_tok": 0.0, "loss_pi": 0.0, "loss_lam": 0.0}
    count = 0

    cpu = torch.device("cpu")
    for bi in range(int(num_batches)):
        rows = [train_ds[i] for i in idxs[bi * batch_size : (bi + 1) * batch_size]]
        x1_ids: list[list[int]] = []
        for r in rows:
            ids = r["input_ids"]
            if not isinstance(ids, list):
                ids = list(ids)
            ids = [int(x) for x in ids]
            if bos_id is not None and ids and ids[0] != bos_id:
                ids = [bos_id] + ids
            x1_ids.append(ids)

        B = len(x1_ids)
        # ---- match trainer perf path: sample τ/t/κ on CPU to avoid device sync ----
        tau_text_cpu = sample_tau_text(
            batch_size=B,
            device=cpu,
            tau_text_max=float(tau_text_max),
        )
        t_text_cpu = tau_to_t_text(tau_text_cpu)
        k_keep_cpu = scheduler.kappa(t_text_cpu).to(cpu)  # [B,1]

        noised = build_noised_xt_and_bags(
            x1_ids=x1_ids,
            kappa_keep=k_keep_cpu,
            device=cpu,
            prompt_len_list=None,
            image_token_id=None,
            disallow_image_in_prompt=True,
        )
        xt_list = noised.xt_list
        bags_list = noised.bags_list

        x_tok, x_mask = pad_1d(xt_list, pad_val=pad_id)
        x_tok = x_tok.to(device)
        x_mask = x_mask.to(device)

        Lmax = int(x_tok.shape[1])
        if bool(condition_text_on_time):
            times = t_text_cpu.to(device).expand(B, Lmax)
        else:
            times = torch.zeros((B, Lmax), device=device, dtype=torch.float32)

        out = model(
            input_ids=x_tok,
            attention_mask=x_mask,
            is_any_modality=torch.zeros_like(x_mask, dtype=torch.bool),
            modality_tokens=None,
            modality_positions=None,
            times=times,
        )
        pi = out["pi"]
        lam = out["lambda_nonzero"]
        q_logits = out["q_logits"]

        logQ = F.log_softmax(q_logits, dim=-1)
        tl = text_loss_paper_eq7_fast(
            pi=pi,
            lam=lam,
            logQ=logQ,
            bags_list=bags_list,
            xt_positions=None,
            normalize_by_n=True,
        )

        sums["loss_total"] += float(tl.total.item())
        sums["loss_tok"] += float(tl.loss_tok.item())
        sums["loss_pi"] += float(tl.loss_pi.item())
        sums["loss_lam"] += float(tl.loss_lam.item())
        count += 1

    return {k: v / float(count) for k, v in sums.items()}


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required")

    device = _resolve_device(args.device)
    tokenizer_dir = args.tokenizer_dir or args.model_dir

    dllm.utils.get_default_logger(__name__).info(
        f"Loading tokenizer from: {tokenizer_dir}\n"
        f"Loading model from: {args.model_dir}\n"
        f"Loading dataset from: {args.dataset_dir}\n"
        f"device={device}"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.pad_token

    ds = load_from_disk(args.dataset_dir)
    if "train" not in ds:
        raise ValueError(f"dataset_dir has no 'train' split: splits={list(ds.keys())}")
    train = ds["train"]
    if "input_ids" not in train.column_names:
        raise ValueError("dataset train split must contain 'input_ids'")

    trained = OneFlowModel.from_pretrained(args.model_dir, map_location="cpu")
    runtime_cfg = load_runtime_config(args.model_dir)
    resolved_eval, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="training",
        cli_overrides={
            "scheduler_cls": args.scheduler_cls,
            "tau_text_max": args.tau_text_max,
            "condition_text_on_time": args.condition_text_on_time,
        },
        defaults=DEFAULT_TEXT_EVAL_RUNTIME,
    )
    if runtime_cfg is None:
        dllm.utils.get_default_logger(__name__).warning(
            "No oneflow_runtime_config.json found in model_dir; falling back to built-in eval defaults."
        )
    if applied_ckpt_keys:
        dllm.utils.get_default_logger(__name__).info(
            f"Using checkpoint runtime config for eval keys: {sorted(applied_ckpt_keys)}"
        )
    if override_keys:
        dllm.utils.get_default_logger(__name__).warning(
            f"CLI overrides checkpoint runtime config for eval keys: {sorted(override_keys)}"
        )

    scheduler_cls = str(resolved_eval["scheduler_cls"])
    tau_text_max = float(resolved_eval["tau_text_max"])
    condition_text_on_time = bool(resolved_eval["condition_text_on_time"])
    seeds = _parse_seeds(int(args.seed), args.seeds)

    trained_per_seed: list[dict[str, float]] = []
    for s in seeds:
        metrics = _eval_model(
            model=trained,
            tokenizer=tokenizer,
            train_ds=train,
            device=device,
            scheduler_cls=scheduler_cls,
            batch_size=int(args.batch_size),
            num_batches=int(args.num_batches),
            seed=int(s),
            condition_text_on_time=condition_text_on_time,
            tau_text_max=tau_text_max,
        )
        trained_per_seed.append({"seed": int(s), **metrics})
    trained_agg = _summarize_metrics(trained_per_seed)

    if len(seeds) == 1:
        single = {k: float(v) for k, v in trained_per_seed[0].items() if k != "seed"}
        print("\n=== Eq7 loss (trained) ===")
        print(json.dumps(single, indent=2))
    else:
        print("\n=== Eq7 loss (trained, aggregate over seeds) ===")
        print(json.dumps(trained_agg, indent=2))

    random_per_seed: list[dict[str, float]] = []
    if bool(args.compare_random):
        cfg_path = os.path.join(args.model_dir, "oneflow_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing oneflow_config.json in model_dir: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = OneFlowConfig(**json.load(f))
        for s in seeds:
            rand_model = OneFlowModel(cfg)
            rand_metrics = _eval_model(
                model=rand_model,
                tokenizer=tokenizer,
                train_ds=train,
                device=device,
                scheduler_cls=scheduler_cls,
                batch_size=int(args.batch_size),
                num_batches=int(args.num_batches),
                seed=int(s),  # same noising RNG indices per seed
                condition_text_on_time=condition_text_on_time,
                tau_text_max=tau_text_max,
            )
            random_per_seed.append({"seed": int(s), **rand_metrics})

        random_agg = _summarize_metrics(random_per_seed)
        if len(seeds) == 1:
            single = {k: float(v) for k, v in random_per_seed[0].items() if k != "seed"}
            print("\n=== Eq7 loss (random init) ===")
            print(json.dumps(single, indent=2))
        else:
            print("\n=== Eq7 loss (random init, aggregate over seeds) ===")
            print(json.dumps(random_agg, indent=2))

    report = {
        "model_dir": args.model_dir,
        "dataset_dir": args.dataset_dir,
        "tokenizer_dir": tokenizer_dir,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "seeds": [int(x) for x in seeds],
        "runtime": {
            "scheduler_cls": scheduler_cls,
            "tau_text_max": tau_text_max,
            "condition_text_on_time": condition_text_on_time,
        },
        "trained": {
            "per_seed": trained_per_seed,
            "aggregate": trained_agg,
        },
    }
    if bool(args.compare_random):
        report["random"] = {
            "per_seed": random_per_seed,
            "aggregate": _summarize_metrics(random_per_seed),
        }
    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] wrote report: {out_path}")


if __name__ == "__main__":
    main()

