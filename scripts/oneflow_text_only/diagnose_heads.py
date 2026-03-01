#!/usr/bin/env python3
"""
Corrected diagnostic script: analyze Q/pi/lambda head statistics and hidden state
diversity from a checkpoint, using **actual noised X_t inputs** (not clean sequences).

Key fix from v1: the model sees X_t sequences that match what it receives during
training at each time value t, with kappa_keep = κ(t) = t.
Always passes times=0 to the model (matching condition_text_on_time=False default).

Reports:
  - Hidden state pairwise cosine similarity (per t)
  - Per-block hidden state cosine similarity (via hooks)
  - Q distribution entropy
  - Q top-k tokens
  - pi / lambda statistics
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import transformers


def pairwise_cosine(h: torch.Tensor) -> float:
    """Compute average pairwise cosine similarity for [L, D] tensor."""
    if h.shape[0] < 2:
        return float("nan")
    h_norm = F.normalize(h.float(), dim=-1)
    cos = h_norm @ h_norm.T  # [L, L]
    L = h.shape[0]
    mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
    return float(cos[mask].mean().item())


def main():
    parser = argparse.ArgumentParser(
        description="Corrected diagnostic: noised X_t inputs + hidden state analysis"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument(
        "--time_values", type=str, default="0.01,0.1,0.3,0.5,0.7,0.9,1.0",
        help="Comma-separated kappa_keep values (= t for LinearKappaScheduler)"
    )
    args = parser.parse_args()

    transformers.set_seed(args.seed)
    device = torch.device(args.device)
    cpu = torch.device("cpu")

    # -- load model --
    from dllm.pipelines.oneflow_text_only.models import OneFlowTextOnlyModel
    from dllm.pipelines.oneflow.sequence_ops import build_noised_xt_and_bags

    print(f"Loading model from {args.model_dir} ...")
    model = OneFlowTextOnlyModel.from_pretrained(args.model_dir, map_location="cpu").eval()
    model = model.to(device)
    n_blocks = model.config.n_blocks
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"  vocab_size={vocab_size}, hidden_size={hidden_size}, "
          f"n_blocks={n_blocks}, n_heads={model.config.n_heads}")

    # -- load tokenizer --
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    print(f"  tokenizer vocab_size={tokenizer.vocab_size}")

    # -- load data (clean x1 sequences) --
    if args.dataset_dir is not None:
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_dir)
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]
        n = min(args.num_samples, len(ds))
        ds = ds.select(range(n))
        samples = []
        for row in ds:
            ids = row.get("input_ids", row.get("token_ids", None))
            if ids is None:
                continue
            ids = list(ids)[:args.max_length]
            samples.append(ids)
            if len(samples) >= args.num_samples:
                break
    else:
        print("  No dataset_dir; using random token IDs.")
        bos = tokenizer.bos_token_id or 0
        samples = []
        for _ in range(args.num_samples):
            length = min(args.max_length, 128)
            ids = [bos] + torch.randint(0, vocab_size, (length,)).tolist()
            samples.append(ids)

    print(f"  Loaded {len(samples)} clean x1 samples (max_length={args.max_length})")

    # -- setup per-block hooks --
    block_outputs: dict[int, torch.Tensor] = {}

    def make_hook(block_idx: int):
        def hook_fn(module, input, output):
            block_outputs[block_idx] = output.detach().float().cpu()
        return hook_fn

    hooks = []
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(make_hook(i)))

    # Also capture input embeddings
    input_embed_capture: list[torch.Tensor] = []

    def embed_hook(module, input, output):
        input_embed_capture.clear()
        input_embed_capture.append(output.detach().float().cpu())

    hooks.append(model.vocab_embed.register_forward_hook(embed_hook))

    time_values = [float(x) for x in args.time_values.split(",")]
    all_results = {}

    for t_val in time_values:
        print(f"\n{'='*60}")
        print(f"  Analyzing at kappa_keep = {t_val} (t = {t_val})")
        print(f"{'='*60}")

        # Accumulators across samples
        all_hidden_cosines = []
        all_block_cosines = defaultdict(list)  # block_idx -> [cosine values]
        all_embed_cosines = []
        all_pi_vals = []
        all_lam_vals = []
        all_q_probs_sum = torch.zeros(vocab_size, dtype=torch.float64)
        total_positions = 0
        all_entropy_vals = []
        all_xt_lengths = []

        for s_idx, x1_ids in enumerate(samples):
            # Apply kappa_keep masking to get X_t
            kappa_keep = torch.tensor([[t_val]])
            noised = build_noised_xt_and_bags(
                x1_ids=[x1_ids],
                kappa_keep=kappa_keep,
                device=cpu,
            )
            xt_ids = noised.xt_list[0]
            all_xt_lengths.append(len(xt_ids))

            if len(xt_ids) < 2:
                # Skip sequences with only BOS (too short for cosine similarity)
                continue

            # Build model input (always times=0, matching training convention)
            input_ids = torch.tensor([xt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            times = torch.zeros_like(input_ids, dtype=torch.float32)

            block_outputs.clear()
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, times=times)

            # -- hidden state cosine similarity --
            hidden = out["hidden_states"][0].float().cpu()  # [L, H]
            cos_val = pairwise_cosine(hidden)
            all_hidden_cosines.append(cos_val)

            # -- per-block cosine similarity --
            for blk_idx in sorted(block_outputs.keys()):
                blk_h = block_outputs[blk_idx][0]  # [L, H]
                all_block_cosines[blk_idx].append(pairwise_cosine(blk_h))

            # -- input embedding cosine --
            if input_embed_capture:
                emb_h = input_embed_capture[0][0]  # [L, H]
                all_embed_cosines.append(pairwise_cosine(emb_h))

            # -- pi / lambda --
            pi = out["pi"][0].float().cpu()  # [L]
            lam = out["lambda_nonzero"][0].float().cpu()  # [L]
            all_pi_vals.append(pi)
            all_lam_vals.append(lam)

            # -- Q statistics --
            q_logits = out["q_logits"][0].float().cpu()  # [L, V]
            q_probs = F.softmax(q_logits, dim=-1)  # [L, V]
            L = q_probs.shape[0]

            # entropy
            log_p = torch.log(q_probs.clamp_min(1e-12))
            ent = -(q_probs * log_p).sum(dim=-1)  # [L]
            all_entropy_vals.append(ent)

            # accumulate for top-k
            all_q_probs_sum += q_probs.sum(dim=0).double()
            total_positions += L

        # -- aggregate results --
        n_valid = len(all_hidden_cosines)
        avg_xt_len = sum(all_xt_lengths) / max(len(all_xt_lengths), 1)
        print(f"\n  X_t stats: avg_length={avg_xt_len:.1f}, "
              f"original_length={sum(len(s) for s in samples)/len(samples):.1f}, "
              f"valid_samples={n_valid}/{len(samples)}")

        if n_valid == 0:
            print("  No valid samples (all too short). Skipping.")
            all_results[f"t={t_val}"] = {"t": t_val, "skipped": True,
                                         "avg_xt_length": avg_xt_len}
            continue

        # Hidden state cosine
        avg_hidden_cos = sum(all_hidden_cosines) / n_valid
        print(f"\n  [Hidden state] avg pairwise cosine = {avg_hidden_cos:.4f}")

        # Input embedding cosine
        avg_embed_cos = sum(all_embed_cosines) / max(len(all_embed_cosines), 1)
        print(f"  [Input embed]  avg pairwise cosine = {avg_embed_cos:.4f}")

        # Per-block cosine
        block_cos_summary = {}
        print(f"\n  [Per-block pairwise cosine similarity]:")
        for blk_idx in sorted(all_block_cosines.keys()):
            vals = all_block_cosines[blk_idx]
            avg_bc = sum(vals) / len(vals)
            block_cos_summary[f"block_{blk_idx}"] = avg_bc
            print(f"    block {blk_idx:2d}: {avg_bc:.4f}")

        # Pi
        all_pi = torch.cat(all_pi_vals)
        pi_mean = float(all_pi.mean().item())
        pi_std = float(all_pi.std().item())
        print(f"\n  [pi] mean={pi_mean:.4f}  std={pi_std:.4f}  "
              f"min={float(all_pi.min().item()):.4f}  max={float(all_pi.max().item()):.4f}")

        # Lambda
        all_lam = torch.cat(all_lam_vals)
        lam_mean = float(all_lam.mean().item())
        lam_std = float(all_lam.std().item())
        print(f"  [lambda] mean={lam_mean:.4f}  std={lam_std:.4f}  "
              f"min={float(all_lam.min().item()):.4f}  max={float(all_lam.max().item()):.4f}")

        # Q entropy
        all_ent = torch.cat(all_entropy_vals)
        max_entropy = float(torch.log(torch.tensor(float(vocab_size))).item())
        ent_mean = float(all_ent.mean().item())
        ent_std = float(all_ent.std().item())
        print(f"  [Q entropy] mean={ent_mean:.4f}  std={ent_std:.4f}  "
              f"(max_possible={max_entropy:.4f})")

        # Q top tokens
        avg_probs = (all_q_probs_sum / max(total_positions, 1)).float()
        top_k = 15
        top_vals, top_ids = avg_probs.topk(top_k)
        print(f"\n  [Q top-{top_k} tokens by average probability]:")
        for rank, (prob, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist())):
            try:
                tok_str = tokenizer.decode([tid])
            except Exception:
                tok_str = f"<id={tid}>"
            print(f"    #{rank+1:2d}: id={tid:6d}  prob={prob:.6f}  token='{tok_str}'")

        top1_prob = float(top_vals[0].item())
        top10_sum = float(top_vals[:10].sum().item())

        result = {
            "t": t_val,
            "avg_xt_length": avg_xt_len,
            "n_valid_samples": n_valid,
            "hidden_state_cosine": avg_hidden_cos,
            "input_embed_cosine": avg_embed_cos,
            "per_block_cosine": block_cos_summary,
            "pi": {"mean": pi_mean, "std": pi_std},
            "lambda": {"mean": lam_mean, "std": lam_std},
            "q_entropy": {"mean": ent_mean, "std": ent_std, "max_possible": max_entropy},
            "q_top_tokens": [
                {"rank": r+1, "token_id": int(tid), "prob": float(p),
                 "token_str": tokenizer.decode([int(tid)])}
                for r, (p, tid) in enumerate(zip(top_vals.tolist(), top_ids.tolist()))
            ],
            "q_concentration": {"top1_prob": top1_prob, "top10_sum": top10_sum},
        }
        all_results[f"t={t_val}"] = result

    # Remove hooks
    for h in hooks:
        h.remove()

    # -- summary table --
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'t':>6} | {'xt_len':>7} | {'embed_cos':>10} | {'hidden_cos':>10} | "
          f"{'Q_entropy':>10} | {'pi_mean':>8} | {'lam_mean':>9} | {'Q_top1':>12}")
    print("-" * 90)
    for t_val in time_values:
        key = f"t={t_val}"
        r = all_results.get(key, {})
        if r.get("skipped"):
            print(f"{t_val:>6.2f} | {'SKIP':>7} | {'---':>10} | {'---':>10} | "
                  f"{'---':>10} | {'---':>8} | {'---':>9} | {'---':>12}")
            continue
        xt_l = r.get("avg_xt_length", 0)
        e_cos = r.get("input_embed_cosine", float("nan"))
        h_cos = r.get("hidden_state_cosine", float("nan"))
        q_ent = r.get("q_entropy", {}).get("mean", float("nan"))
        pi_m = r.get("pi", {}).get("mean", float("nan"))
        lam_m = r.get("lambda", {}).get("mean", float("nan"))
        top1 = r.get("q_top_tokens", [{}])[0].get("token_str", "?") if r.get("q_top_tokens") else "?"
        top1_p = r.get("q_concentration", {}).get("top1_prob", float("nan"))
        print(f"{t_val:>6.2f} | {xt_l:>7.1f} | {e_cos:>10.4f} | {h_cos:>10.4f} | "
              f"{q_ent:>10.4f} | {pi_m:>8.4f} | {lam_m:>9.4f} | {top1_p:.4f} '{top1}'")

    # -- save --
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(args.model_dir, "diagnose_heads_v2.json")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
