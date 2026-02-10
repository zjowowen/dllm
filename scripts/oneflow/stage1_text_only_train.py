"""
Stage 1 (text-only) smoke training script.

This script is meant to validate the "## Stage 1：纯文本（text-only）正确性" section in:
  doc/oneflow/validation/oneflow_zero_validation_zh.md

It exercises (text-only; no image/latent dependency):
  - τ_text sampling -> t_text clipping
  - κ(t_text) keep-prob
  - X_t + bag-of-tokens A_i construction (build_noised_xt_and_bags)
  - Eq(7) paper loss: token CE + π BCE + λ_nonzero Poisson(k>0)
  - backward + optimizer step on a tiny toy model

Run (recommended via wrapper):
  bash scripts/oneflow/stage1_text_only_train.sh --max_steps 20 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.schedulers import CubicKappaScheduler
from dllm.pipelines.ctmc_utils import pad_1d
from dllm.pipelines.oneflow.losses import TextEq7Loss, text_loss_paper_eq7
from dllm.pipelines.oneflow.sequence_ops import (
    build_noised_xt_and_bags,
    sample_tau_text,
    tau_to_t_text,
)


def _resolve_device(name: str) -> torch.device:
    name = str(name or "cpu").strip().lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if name == "npu":
        # torch-npu uses device string "npu"
        return torch.device("npu")
    if name == "cpu":
        return torch.device("cpu")
    # fallback: allow things like "cuda:0", "npu:0"
    return torch.device(name)


@dataclass
class StepTrace:
    step: int
    tau_text: list[float]
    t_text: list[float]
    kappa_keep: list[float]
    xt_list: list[list[int]]
    bags_list: list[list[list[int]]]
    loss_total: float
    loss_tok: float
    loss_pi: float
    loss_lam: float


class ToyTextOnlyModel(nn.Module):
    """
    A tiny per-token model that produces (pi, lambda_nonzero, q_logits) for Eq(7).

    This intentionally avoids OneFlowModel's Transfusion dependency; it only aims to
    validate Stage-1 discrete logic + loss wiring + backward.
    """

    def __init__(self, *, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)

        self.text_embed = nn.Embedding(self.vocab_size, self.dim)
        self.time_proj = nn.Linear(1, self.dim)

        self.to_pi = nn.Linear(self.dim, 1)
        self.to_lam = nn.Linear(self.dim, 1)
        self.to_q = nn.Linear(self.dim, self.vocab_size)

    def forward(self, *, input_ids: torch.Tensor, times: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,L], got {tuple(input_ids.shape)}")
        if times.shape != input_ids.shape:
            raise ValueError(f"times must match input_ids shape, got {tuple(times.shape)}")

        ids = input_ids.clamp_min(0)
        h = self.text_embed(ids)
        h = h + self.time_proj(times.unsqueeze(-1))
        h = torch.tanh(h)

        pi = torch.sigmoid(self.to_pi(h)).squeeze(-1)
        lam = F.softplus(self.to_lam(h)).squeeze(-1)
        q_logits = self.to_q(h)
        return {"pi": pi, "lambda_nonzero": lam, "q_logits": q_logits}


def _assert_stage1_invariants(*, x1_ids: list[list[int]], xt_list: list[list[int]], bags_list: list[list[list[int]]]):
    if len(x1_ids) != len(xt_list) or len(x1_ids) != len(bags_list):
        raise AssertionError("Batch size mismatch among x1_ids/xt_list/bags_list.")
    for b, (x1, xt, bags) in enumerate(zip(x1_ids, xt_list, bags_list)):
        if not x1 or not xt:
            raise AssertionError(f"Empty sequence found at sample {b}.")
        if xt[0] != x1[0]:
            raise AssertionError(f"Sample {b}: BOS mismatch: xt[0]={xt[0]} vs x1[0]={x1[0]}.")
        if len(bags) != len(xt):
            raise AssertionError(f"Sample {b}: len(bags)={len(bags)} != len(xt)={len(xt)}.")
        deleted = sum(len(bag) for bag in bags)
        if deleted != (len(x1) - len(xt)):
            raise AssertionError(
                f"Sample {b}: deletion conservation failed: sum|A_i|={deleted} "
                f"but len(x1)-len(xt)={len(x1)-len(xt)}."
            )


def _format_position_table(*, xt: list[int], bags: list[list[int]]) -> str:
    # Simple markdown table for quick eyeballing (ids only).
    lines = []
    lines.append("| i(slot) | X_t[i] token_id | bag A_i (token_ids) | k_i |")
    lines.append("|---:|---:|---|---:|")
    for i, (tok, bag) in enumerate(zip(xt, bags)):
        lines.append(f"| {i} | {tok} | {bag} | {len(bag)} |")
    return "\n".join(lines)


def _loss_to_floats(tl: TextEq7Loss) -> tuple[float, float, float, float]:
    return (
        float(tl.total.detach().item()),
        float(tl.loss_tok.detach().item()),
        float(tl.loss_pi.detach().item()),
        float(tl.loss_lam.detach().item()),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data/vis/stage1_text_only_smoke")
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "npu"])

    # synthetic text-only batch
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=12, help="includes BOS+EOS")
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--prompt_len", type=int, default=2, help="0 disables prompt forcing")

    # scheduler / τ_text sampling
    p.add_argument("--fixed_tau_text", type=float, default=0.2, help="set <0 to resample each step")
    p.add_argument(
        "--tau_text_max",
        type=float,
        default=2.0,
        help="Upper bound for τ_text sampling when resampling (default: 2.0).",
    )
    p.add_argument(
        "--resample_noising_each_step",
        action="store_true",
        help="If set, rebuild X_t/bags every step (Algorithm-3 style).",
    )
    p.add_argument(
        "--condition_text_on_time",
        action="store_true",
        help="If set, use t_text as model per-token time; otherwise use constant 0 (paper default).",
    )

    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()

    device = _resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(int(args.seed))

    B = int(args.batch_size)
    seq_len = int(args.seq_len)
    vocab_size = int(args.vocab_size)
    if seq_len < 2:
        raise ValueError("--seq_len must be >= 2 (BOS+EOS).")
    if vocab_size <= 4:
        raise ValueError("--vocab_size must be > 4 (need room for special+random tokens).")

    pad_id = 0
    bos_id = 1
    eos_id = 2

    # ---- build a synthetic text-only batch (python lists of ids) ------------------
    # ids in [3, vocab_size) to avoid clashing with PAD/BOS/EOS
    mid = torch.randint(low=3, high=vocab_size, size=(B, seq_len - 2), dtype=torch.long)
    x1_ids = [[bos_id] + mid[b].tolist() + [eos_id] for b in range(B)]

    prompt_len_list = None
    if int(args.prompt_len) > 0:
        pl = int(args.prompt_len)
        if pl > seq_len:
            raise ValueError(f"--prompt_len={pl} exceeds --seq_len={seq_len}.")
        prompt_len_list = [pl for _ in range(B)]

    sched = CubicKappaScheduler()

    model = ToyTextOnlyModel(vocab_size=vocab_size, dim=int(args.dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    traces: list[StepTrace] = []

    # optional: build fixed noising once to make the smoke deterministic & stable
    fixed = float(args.fixed_tau_text)
    fixed_tau: torch.Tensor | None = None
    fixed_t: torch.Tensor | None = None
    fixed_k: torch.Tensor | None = None
    fixed_xt: list[list[int]] | None = None
    fixed_bags: list[list[list[int]]] | None = None

    if fixed >= 0.0 and (not bool(args.resample_noising_each_step)):
        fixed_tau = torch.full((B, 1), float(fixed), device=device, dtype=torch.float32)
        fixed_t = tau_to_t_text(fixed_tau)
        fixed_k = sched.kappa(fixed_t).to(device)
        noised = build_noised_xt_and_bags(
            x1_ids=x1_ids,
            kappa_keep=fixed_k,
            device=device,
            prompt_len_list=prompt_len_list,
            image_token_id=None,
            disallow_image_in_prompt=True,
        )
        fixed_xt, fixed_bags = noised.xt_list, noised.bags_list
        _assert_stage1_invariants(x1_ids=x1_ids, xt_list=fixed_xt, bags_list=fixed_bags)

    # ---- training loop ------------------------------------------------------------
    for step in range(int(args.max_steps)):
        model.train()
        opt.zero_grad(set_to_none=True)

        if fixed_xt is not None and fixed_bags is not None and fixed_tau is not None and fixed_t is not None:
            tau_text = fixed_tau
            t_text = fixed_t
            kappa_keep = fixed_k if fixed_k is not None else sched.kappa(t_text).to(device)
            xt_list = fixed_xt
            bags_list = fixed_bags
        else:
            if fixed >= 0.0:
                tau_text = torch.full((B, 1), float(fixed), device=device, dtype=torch.float32)
            else:
                tau_text = sample_tau_text(
                    batch_size=B,
                    device=device,
                    tau_text_max=float(args.tau_text_max),
                )
            t_text = tau_to_t_text(tau_text)
            kappa_keep = sched.kappa(t_text).to(device)

            noised = build_noised_xt_and_bags(
                x1_ids=x1_ids,
                kappa_keep=kappa_keep,
                device=device,
                prompt_len_list=prompt_len_list,
                image_token_id=None,
                disallow_image_in_prompt=True,
            )
            xt_list, bags_list = noised.xt_list, noised.bags_list
            _assert_stage1_invariants(x1_ids=x1_ids, xt_list=xt_list, bags_list=bags_list)

        # pad X_t for per-token model forward
        x_tok, x_mask = pad_1d(xt_list, pad_val=pad_id)  # [B,L], [B,L]
        x_tok = x_tok.to(device)

        Lmax = int(x_tok.shape[1])
        if bool(args.condition_text_on_time):
            times = t_text.expand(B, Lmax)
        else:
            times = torch.zeros((B, Lmax), device=device, dtype=torch.float32)

        out = model(input_ids=x_tok, times=times)
        logQ = F.log_softmax(out["q_logits"], dim=-1)
        tl = text_loss_paper_eq7(
            pi=out["pi"],
            lam=out["lambda_nonzero"],
            logQ=logQ,
            bags_list=bags_list,
            xt_positions=None,
            normalize_by_n=True,
        )
        loss = tl.total
        loss.backward()
        opt.step()

        total, tok, pi, lam = _loss_to_floats(tl)
        traces.append(
            StepTrace(
                step=int(step),
                tau_text=[float(x) for x in tau_text.detach().reshape(-1).tolist()],
                t_text=[float(x) for x in t_text.detach().reshape(-1).tolist()],
                kappa_keep=[float(x) for x in kappa_keep.detach().reshape(-1).tolist()],
                xt_list=xt_list,
                bags_list=bags_list,
                loss_total=total,
                loss_tok=tok,
                loss_pi=pi,
                loss_lam=lam,
            )
        )

        if step == 0:
            # Print a position table for sample0 once (helpful for eyeballing Stage-1 invariants).
            print("\n[Stage1] sample0 position table:")
            print(_format_position_table(xt=xt_list[0], bags=bags_list[0]))
            print("")

        print(
            f"[Stage1][step={step:04d}] "
            f"loss={total:.6f} (tok={tok:.6f}, pi={pi:.6f}, lam={lam:.6f}) "
            f"t_text={traces[-1].t_text} kappa_keep={traces[-1].kappa_keep}"
        )

    out_path = os.path.join(args.output_dir, "stage1_text_only_trace.json")
    payload: dict[str, Any] = {
        "args": vars(args),
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "x1_ids": x1_ids,
        "traces": [asdict(t) for t in traces],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Stage1] Wrote trace: {out_path}")


if __name__ == "__main__":
    main()

