#!/usr/bin/env python3
"""
Batch prompt evaluation for oneflow_text_only pipeline.

Outputs:
  - <output_dir>/report.jsonl
  - <output_dir>/report.md
  - (optional) <output_dir>/visual/<prompt_id>/visual_rank*_sample*.{html,json}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

import dllm
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)
from dllm.pipelines.oneflow_text_only.models import OneFlowTextOnlyModel
from dllm.pipelines.oneflow_text_only.sampler import (
    OneFlowTextOnlySampler,
    OneFlowTextOnlySamplerConfig,
)
from dllm.pipelines.oneflow_text_only.visualize import (
    build_intermediates_from_histories,
    render_rich_timeline,
    save_visual_artifacts,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ScriptArguments:
    model_dir: str = ""
    prompts_file: str = "scripts/oneflow/eval_prompts_text_minimal.jsonl"
    output_dir: str = "outputs/oneflow_text_only_prompt_eval"
    seed: int = 42
    device: str = "auto"  # auto|cpu|cuda|npu
    skip_special_tokens: bool = True
    max_prompts: int = 0  # <=0 means all

    # visualize (visually_generate-style)
    visualize: bool = False
    visualize_terminal: bool = True
    visualize_save_html: bool = True
    visualize_save_json: bool = True
    visualize_color_map: str = "jet"  # jet|rainbow|viridis|black_to_lightblue
    visualize_terminal_replay: bool = False
    visualize_replay_step_delay_sec: float = 0.25
    visualize_replay_hold_sec: float = 5.0
    visualize_replay_until_interrupt: bool = False


@dataclass
class SamplerArgs(OneFlowTextOnlySamplerConfig):
    dt: float | None = None
    max_steps: int | None = None
    image_num_tokens: int = 0
    temperature: float | None = None
    use_pi_gate: bool | None = None
    max_w: float | None = None
    condition_text_on_time: bool | None = None
    kappa_scheduler_cls: str | None = None
    return_dict: bool = True


def _npu_available() -> bool:
    return bool(hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available())


def _resolve_device(name: str) -> torch.device:
    dev = str(name or "auto").lower()
    if dev == "auto":
        if _npu_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if dev in ("cpu", "cuda", "npu"):
        return torch.device(dev)
    raise ValueError(f"Unknown --device: {name} (expected auto|cpu|cuda|npu)")


def _load_prompts(path: str, max_prompts: int) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if "prompt" not in row:
                raise ValueError(f"Line {i + 1} has no 'prompt' field.")
            row.setdefault("id", f"row_{i + 1}")
            row.setdefault("category", "general")
            out.append(row)
            if int(max_prompts) > 0 and len(out) >= int(max_prompts):
                break
    return out


def _contains_checks(text: str, expected_contains) -> tuple[list[dict], bool | None]:
    if expected_contains is None:
        return [], None
    if isinstance(expected_contains, str):
        needles = [expected_contains]
    elif isinstance(expected_contains, list):
        needles = [str(x) for x in expected_contains]
    else:
        raise ValueError("expected_contains must be string, list[string], or null")
    checks = []
    ok_all = True
    low = text.lower()
    for n in needles:
        ok = str(n).lower() in low
        checks.append({"needle": str(n), "ok": bool(ok)})
        ok_all = bool(ok_all and ok)
    return checks, bool(ok_all)


def _build_markdown(records: list[dict], model_dir: str, prompts_file: str) -> str:
    total = len(records)
    has_expect = [r for r in records if r.get("expected_pass") is not None]
    pass_cnt = sum(1 for r in has_expect if bool(r.get("expected_pass")))

    lines = []
    lines.append("# OneFlow Text-Only Prompt Eval Report")
    lines.append("")
    lines.append(f"- model_dir: `{model_dir}`")
    lines.append(f"- prompts_file: `{prompts_file}`")
    lines.append(f"- total_prompts: {total}")
    if has_expect:
        lines.append(f"- expected_contains_pass: {pass_cnt}/{len(has_expect)}")
    lines.append("")
    lines.append("| id | category | expected_pass | output_preview | visual_dir |")
    lines.append("|---|---|---:|---|---|")
    for r in records:
        p = r.get("expected_pass")
        ps = "N/A" if p is None else ("1" if bool(p) else "0")
        preview = str(r.get("output_text", "")).replace("\n", " ").strip()
        if len(preview) > 100:
            preview = preview[:97] + "..."
        vis_dir = str(r.get("visual_dir", ""))
        lines.append(f"| {r.get('id')} | {r.get('category')} | {ps} | {preview} | {vis_dir} |")
    lines.append("")
    return "\n".join(lines)


def _safe_id(raw: str) -> str:
    out = []
    for c in str(raw):
        if c.isalnum() or c in ("-", "_"):
            out.append(c)
        else:
            out.append("_")
    return "".join(out)[:120] or "sample"


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerArgs))
    script_args, sampler_args = parser.parse_args_into_dataclasses()
    if not script_args.model_dir:
        raise ValueError("--model_dir is required")

    transformers.set_seed(int(script_args.seed))
    os.makedirs(script_args.output_dir, exist_ok=True)

    prompts = _load_prompts(script_args.prompts_file, int(script_args.max_prompts))
    if not prompts:
        raise ValueError(f"No prompts found in: {script_args.prompts_file}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_dir)
    model = OneFlowTextOnlyModel.from_pretrained(script_args.model_dir, map_location="cpu").eval()
    device = _resolve_device(script_args.device)
    model = model.to(device)
    sampler = OneFlowTextOnlySampler(model=model, tokenizer=tokenizer)

    runtime_cfg = load_runtime_config(script_args.model_dir)
    resolved_sampling, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="sampling",
        cli_overrides={
            "scheduler_cls": sampler_args.kappa_scheduler_cls,
            "dt": sampler_args.dt,
            "max_steps": sampler_args.max_steps,
            "temperature": sampler_args.temperature,
            "use_pi_gate": sampler_args.use_pi_gate,
            "append_only": sampler_args.append_only,
            "edit_prompt": sampler_args.edit_prompt,
            "condition_text_on_time": sampler_args.condition_text_on_time,
            "max_w": sampler_args.max_w,
        },
        defaults=DEFAULT_TEXT_SAMPLER_RUNTIME,
    )
    if runtime_cfg is None:
        logger.warning("No oneflow_runtime_config.json found in model_dir; using built-in sampler defaults.")
    if applied_ckpt_keys:
        logger.info(f"Using checkpoint runtime config for sampler keys: {sorted(applied_ckpt_keys)}")
    if override_keys:
        logger.warning(f"CLI overrides checkpoint runtime config for sampler keys: {sorted(override_keys)}")

    sampler_args.kappa_scheduler_cls = str(resolved_sampling["scheduler_cls"])
    sampler_args.dt = float(resolved_sampling["dt"])
    sampler_args.max_steps = int(resolved_sampling["max_steps"])
    sampler_args.temperature = float(resolved_sampling["temperature"])
    sampler_args.use_pi_gate = bool(resolved_sampling["use_pi_gate"])
    sampler_args.append_only = bool(resolved_sampling["append_only"])
    sampler_args.edit_prompt = bool(resolved_sampling["edit_prompt"])
    sampler_args.condition_text_on_time = bool(resolved_sampling["condition_text_on_time"])
    sampler_args.max_w = None if resolved_sampling["max_w"] is None else float(resolved_sampling["max_w"])

    out_jsonl = os.path.join(script_args.output_dir, "report.jsonl")
    out_md = os.path.join(script_args.output_dir, "report.md")
    records: list[dict] = []

    with open(out_jsonl, "w", encoding="utf-8") as fj:
        for i, row in enumerate(prompts):
            prompt = str(row["prompt"])
            pid = str(row.get("id", f"row_{i + 1}"))
            cat = str(row.get("category", "general"))

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            out = sampler.sample([prompt_ids], sampler_args, return_dict=True)

            text = tokenizer.decode(
                out.sequences[0].tolist(),
                skip_special_tokens=bool(getattr(script_args, "skip_special_tokens", True)),
            )
            checks, expected_pass = _contains_checks(text, row.get("expected_contains", None))

            visual_dir = ""
            if bool(script_args.visualize):
                try:
                    histories = out.histories or [out.sequences]
                    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
                    intermediates, time_grid, valid_lengths = build_intermediates_from_histories(
                        histories=histories,
                        pad_token_id=pad_id,
                        device=torch.device("cpu"),
                    )
                    visual_dir_path = Path(script_args.output_dir) / "visual" / f"{i:04d}_{_safe_id(pid)}"
                    visual_dir = str(visual_dir_path)
                    if bool(script_args.visualize_terminal):
                        render_rich_timeline(
                            intermediates=intermediates,
                            time_grid=time_grid,
                            tokenizer=tokenizer,
                            sample_indices=[0],
                            valid_lengths=valid_lengths,
                            color_map=str(script_args.visualize_color_map),
                            terminal_replay=bool(script_args.visualize_terminal_replay),
                            replay_step_delay_sec=float(script_args.visualize_replay_step_delay_sec),
                            replay_hold_sec=float(script_args.visualize_replay_hold_sec),
                            replay_until_interrupt=bool(script_args.visualize_replay_until_interrupt),
                        )
                    if bool(script_args.visualize_save_html) or bool(script_args.visualize_save_json):
                        save_visual_artifacts(
                            intermediates=intermediates,
                            time_grid=time_grid,
                            tokenizer=tokenizer,
                            sample_indices=[0],
                            valid_lengths=valid_lengths,
                            output_dir=visual_dir_path,
                            rank=0,
                            step=i + 1,
                            save_html=bool(script_args.visualize_save_html),
                            save_json=bool(script_args.visualize_save_json),
                            color_map=str(script_args.visualize_color_map),
                        )
                except ImportError as err:
                    logger.warning(f"Visualization requested but unavailable: {err}")

            rec = {
                "id": pid,
                "category": cat,
                "prompt": prompt,
                "output_text": text,
                "expected_contains": row.get("expected_contains", None),
                "expected_checks": checks,
                "expected_pass": expected_pass,
                "output_num_tokens": int(out.sequences[0].numel()),
                "visual_dir": visual_dir,
            }
            records.append(rec)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fj.flush()
            logger.info(f"[{i + 1}/{len(prompts)}] {pid} ({cat}) done.")

    md = _build_markdown(records, script_args.model_dir, script_args.prompts_file)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Wrote JSONL report: {out_jsonl}")
    logger.info(f"Wrote Markdown report: {out_md}")


if __name__ == "__main__":
    main()

