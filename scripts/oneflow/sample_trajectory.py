"""
OneFlow 采样轨迹可视化脚本。

逐步展示 insertion-based 采样过程：每一步的 t_text、序列长度、
新增 token、以及当前序列的文本内容。

输出：
  - trajectory.jsonl: 每行一个 step 的详细信息（t_text, len, insertions, text）
  - trajectory_summary.txt: 人类可读的逐步摘要
  - final.txt: 最终生成文本

用法:
  python scripts/oneflow/sample_trajectory.py \
    --model_dir data/ckpts/text_pt_overfit_12k_16npu/checkpoint-15000 \
    --dataset_dir data/offline/pt_text_dclm_1024_12k/dataset \
    --tokenizer_dir data/offline/pt_text_dclm_1024_12k/tokenizer \
    --device npu \
    --sample_index 0 \
    --prefix_tokens 10 \
    --dt 0.05 \
    --max_steps 200 \
    --temperature 0.9
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import transformers
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)


def _resolve_device(name: str) -> torch.device:
    name = str(name or "auto").lower()
    if name == "auto":
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


@dataclass
class Args:
    model_dir: str = ""
    dataset_dir: str = ""
    tokenizer_dir: str | None = None
    device: str = "auto"

    sample_index: int = 0
    prefix_tokens: int = 10
    ref_tokens: int = 50

    # sampling params
    seed: int = 42
    dt: float | None = None
    max_steps: int | None = None
    temperature: float | None = None
    use_pi_gate: bool | None = None
    append_only: bool | None = None
    max_new_tokens: int | None = 512
    max_seq_len: int | None = 1024
    max_insertions_per_step: int | None = 64
    max_w: float | None = None
    condition_text_on_time: bool | None = None
    kappa_scheduler_cls: str | None = None

    output_dir: str = "data/vis/sample_trajectory"


def main():
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.model_dir:
        raise ValueError("--model_dir is required")
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required")

    device = _resolve_device(args.device)

    from datasets import load_from_disk
    from dllm.pipelines.oneflow.models import OneFlowModel
    from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput

    os.makedirs(args.output_dir, exist_ok=True)

    tok_dir = args.tokenizer_dir or args.model_dir
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_dir)

    ds = load_from_disk(args.dataset_dir)
    train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds

    idx = int(args.sample_index)
    ids = [int(x) for x in train[idx]["input_ids"]]
    prefix_n = max(1, int(args.prefix_tokens))
    ref_n = max(1, int(args.ref_tokens))

    prefix_ids = ids[:prefix_n]
    ref_ids = ids[prefix_n : prefix_n + ref_n]

    prompt_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=False)

    print(f"=== Prompt ({prefix_n} tokens) ===")
    print(prompt_text)
    print(f"\n=== Reference continuation ({ref_n} tokens) ===")
    print(ref_text)
    print()

    model = OneFlowModel.from_pretrained(args.model_dir, map_location="cpu").eval().to(device)
    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)

    runtime_cfg = load_runtime_config(args.model_dir)
    resolved_sampling, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="sampling",
        cli_overrides={
            "scheduler_cls": args.kappa_scheduler_cls,
            "dt": args.dt,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "use_pi_gate": args.use_pi_gate,
            "append_only": args.append_only,
            "condition_text_on_time": args.condition_text_on_time,
            "max_w": args.max_w,
        },
        defaults=DEFAULT_TEXT_SAMPLER_RUNTIME,
    )
    if runtime_cfg is None:
        print("[warn] no oneflow_runtime_config.json found; using built-in sampler defaults")
    if applied_ckpt_keys:
        print(f"[info] using checkpoint runtime config keys: {sorted(applied_ckpt_keys)}")
    if override_keys:
        print(f"[warn] CLI overrides checkpoint runtime config keys: {sorted(override_keys)}")

    dt = float(resolved_sampling["dt"])
    max_steps = int(resolved_sampling["max_steps"])
    temperature = float(resolved_sampling["temperature"])
    use_pi_gate = bool(resolved_sampling["use_pi_gate"])
    append_only = bool(resolved_sampling["append_only"])
    condition_text_on_time = bool(resolved_sampling["condition_text_on_time"])
    max_w = None if resolved_sampling["max_w"] is None else float(resolved_sampling["max_w"])
    kappa_scheduler_cls = str(resolved_sampling["scheduler_cls"])

    cfg = OneFlowSamplerConfig(
        dt=dt,
        max_steps=max_steps,
        temperature=temperature,
        use_pi_gate=use_pi_gate,
        append_only=append_only,
        max_new_tokens=int(args.max_new_tokens) if args.max_new_tokens else None,
        max_seq_len=int(args.max_seq_len) if args.max_seq_len else None,
        max_insertions_per_step=int(args.max_insertions_per_step) if args.max_insertions_per_step else None,
        max_w=max_w,
        condition_text_on_time=condition_text_on_time,
        kappa_scheduler_cls=kappa_scheduler_cls,
        image_num_tokens=0,
        return_dict=True,
    )

    transformers.set_seed(int(args.seed))
    out = sampler.sample([prefix_ids], cfg, return_dict=True)
    assert isinstance(out, OneFlowSamplerOutput)

    histories = out.histories or []
    final_ids = [int(x) for x in out.sequences[0].detach().to("cpu").tolist()]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=False)

    # ---- Build trajectory ----
    dt = float(dt)
    time_eps = 1e-3
    trajectory = []

    prev_ids: list[int] = list(prefix_ids)
    for step_i, hist_tensor in enumerate(histories):
        cur_ids = [int(x) for x in hist_tensor[0].detach().to("cpu").tolist()]
        t_text = min(1.0, (step_i + 1) * dt)

        # Find insertions (new tokens not in prev)
        new_tokens_count = len(cur_ids) - len(prev_ids)
        # Decode new tokens by diffing
        inserted_texts = []
        if new_tokens_count > 0:
            # Simple heuristic: find tokens in cur but not positionally in prev
            # Use set difference on (position-aware) comparison
            prev_set = set(range(len(prev_ids)))
            j = 0
            new_positions = []
            for i, tok in enumerate(cur_ids):
                if j < len(prev_ids) and tok == prev_ids[j]:
                    j += 1
                else:
                    new_positions.append(i)
            for pos in new_positions:
                tok_id = cur_ids[pos]
                tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
                inserted_texts.append({"position": pos, "token_id": tok_id, "token": tok_str})

        cur_text = tokenizer.decode(cur_ids, skip_special_tokens=False)

        entry = {
            "step": step_i + 1,
            "t_text": round(t_text, 4),
            "seq_len": len(cur_ids),
            "new_tokens": new_tokens_count,
            "insertions": inserted_texts,
            "text": cur_text,
        }
        trajectory.append(entry)
        prev_ids = cur_ids

        if t_text >= 1.0 - time_eps:
            break

    # ---- Write outputs ----
    # 1. JSONL trajectory
    jsonl_path = os.path.join(args.output_dir, "trajectory.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in trajectory:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. Human-readable summary
    summary_path = os.path.join(args.output_dir, "trajectory_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt ({prefix_n} tokens): {prompt_text}\n")
        f.write(f"Reference ({ref_n} tokens): {ref_text}\n")
        f.write("=" * 80 + "\n\n")

        for entry in trajectory:
            if entry["new_tokens"] > 0:
                ins_str = ", ".join(
                    f"pos={ins['position']} '{ins['token']}'"
                    for ins in entry["insertions"][:10]  # cap display
                )
                if len(entry["insertions"]) > 10:
                    ins_str += f" ... (+{len(entry['insertions'])-10} more)"
                f.write(
                    f"[Step {entry['step']:>3d}] t={entry['t_text']:.2f}  "
                    f"len={entry['seq_len']:>4d}  +{entry['new_tokens']} tokens: {ins_str}\n"
                )
            # Print full text every 10 steps or on last step
            if entry["step"] % 10 == 0 or entry == trajectory[-1]:
                f.write(f"  >>> {entry['text'][:200]}{'...' if len(entry['text'])>200 else ''}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Final ({len(final_ids)} tokens): {final_text}\n")

    # 3. Final text
    final_path = os.path.join(args.output_dir, "final.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    # 4. Summary JSON
    meta = {
        "model_dir": args.model_dir,
        "sample_index": idx,
        "prefix_tokens": prefix_n,
        "total_steps": len(trajectory),
        "final_seq_len": len(final_ids),
        "total_inserted": len(final_ids) - prefix_n,
        "steps_with_insertions": sum(1 for e in trajectory if e["new_tokens"] > 0),
        "dt": dt,
        "temperature": temperature,
        "scheduler_cls": kappa_scheduler_cls,
        "condition_text_on_time": condition_text_on_time,
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, ensure_ascii=False, indent=2, fp=f)

    # ---- Print summary to console ----
    print(f"=== Sampling Trajectory Summary ===")
    print(f"  Steps taken:    {len(trajectory)}")
    print(f"  Final seq len:  {len(final_ids)} ({len(final_ids) - prefix_n} inserted)")
    print(f"  Steps w/ inserts: {meta['steps_with_insertions']}")
    print()

    # Show steps with insertions
    for entry in trajectory:
        if entry["new_tokens"] > 0:
            ins_preview = ", ".join(
                f"'{ins['token'].strip()}'" for ins in entry["insertions"][:5]
            )
            if len(entry["insertions"]) > 5:
                ins_preview += f" +{len(entry['insertions'])-5} more"
            print(f"  Step {entry['step']:>3d} (t={entry['t_text']:.2f}): +{entry['new_tokens']} tokens [{ins_preview}]")

    print(f"\n=== Final text (first 300 chars) ===")
    print(final_text[:300])

    print(f"\nSaved to {args.output_dir}/:")
    print(f"  trajectory.jsonl         -- per-step details")
    print(f"  trajectory_summary.txt   -- human-readable summary")
    print(f"  final.txt                -- final generated text")
    print(f"  meta.json                -- run metadata")


if __name__ == "__main__":
    main()
