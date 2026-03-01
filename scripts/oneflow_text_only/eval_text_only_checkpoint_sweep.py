#!/usr/bin/env python3
"""
Stable checkpoint sweep for oneflow_text_only checkpoints.

For each checkpoint under --ckpt_root:
1) Runs multi-seed loss evaluation.
2) Optionally runs batch prompt evaluation.
3) Produces ranked summary (JSON + Markdown) for checkpoint selection.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import transformers


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOSS_EVAL_SCRIPT = os.path.join(REPO_ROOT, "scripts", "oneflow_text_only", "eval_text_only_loss.py")
PROMPT_EVAL_SCRIPT = os.path.join(REPO_ROOT, "scripts", "oneflow_text_only", "eval_text_only_prompts.py")


@dataclass
class Args:
    ckpt_root: str = ""
    dataset_dir: str = ""
    prompts_file: str = "scripts/oneflow/eval_prompts_text_tiered_v1.jsonl"
    output_dir: str = ""

    # Loss eval options
    device: str = "auto"
    loss_batch_size: int = 8
    loss_num_batches: int = 100
    loss_seed: int = 42
    loss_seeds: str = "41,42,43"
    loss_compare_random: bool = False

    # Prompt eval options
    run_prompt_eval: bool = True
    prompt_device: str = ""
    prompt_seed: int = 42
    prompt_max_prompts: int = 0

    # Optional visualization during prompt eval
    prompt_visualize: bool = False
    prompt_visualize_terminal: bool = False
    prompt_visualize_save_html: bool = True
    prompt_visualize_save_json: bool = True
    prompt_visualize_color_map: str = "jet"

    # Checkpoint filtering
    include_final: bool = True
    min_step: int = 0
    max_step: int = 0

    # Ranking mode: hybrid | loss_tok | loss_total | prompt_first
    ranking_mode: str = "hybrid"


def _abs_path(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(REPO_ROOT, path))


def _parse_step_from_name(name: str) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def _discover_checkpoints(args: Args) -> list[dict]:
    root = _abs_path(args.ckpt_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"--ckpt_root not found: {root}")

    out: list[dict] = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        name = entry.name
        step = _parse_step_from_name(name)
        if step is not None:
            if int(args.min_step) > 0 and step < int(args.min_step):
                continue
            if int(args.max_step) > 0 and step > int(args.max_step):
                continue
            out.append(
                {
                    "name": name,
                    "path": os.path.abspath(entry.path),
                    "step": int(step),
                    "is_final": False,
                }
            )
        elif bool(args.include_final) and name == "checkpoint-final":
            out.append(
                {
                    "name": name,
                    "path": os.path.abspath(entry.path),
                    "step": None,
                    "is_final": True,
                }
            )

    def _sort_key(row: dict) -> tuple[int, int]:
        if row["step"] is None:
            return (1, 10**18)
        return (0, int(row["step"]))

    out.sort(key=_sort_key)
    return out


def _run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load_prompt_pass_stats(report_jsonl: str) -> dict:
    total_rows = 0
    expected_rows = 0
    pass_rows = 0
    with open(report_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            total_rows += 1
            expected_pass = row.get("expected_pass")
            if expected_pass is None:
                continue
            expected_rows += 1
            if bool(expected_pass):
                pass_rows += 1
    pass_rate = None
    if expected_rows > 0:
        pass_rate = float(pass_rows) / float(expected_rows)
    return {
        "total_rows": int(total_rows),
        "expected_rows": int(expected_rows),
        "pass_rows": int(pass_rows),
        "expected_pass_rate": pass_rate,
    }


def _record_sort_key(record: dict, ranking_mode: str) -> tuple:
    loss = record["loss"]["trained"]["aggregate"]
    loss_tok = float(loss["loss_tok"]["mean"])
    loss_total = float(loss["loss_total"]["mean"])
    prompt_rate = record["prompt"]["expected_pass_rate"]

    mode = str(ranking_mode).strip().lower()
    if mode == "loss_tok":
        return (loss_tok, loss_total)
    if mode == "loss_total":
        return (loss_total, loss_tok)
    if mode == "prompt_first":
        score = -1.0 if prompt_rate is None else float(prompt_rate)
        return (-score, loss_tok, loss_total)
    if mode != "hybrid":
        raise ValueError(
            f"Unknown --ranking_mode={ranking_mode} "
            "(expected hybrid|loss_tok|loss_total|prompt_first)"
        )
    # hybrid: prioritize prompt pass-rate when available, then tok-loss
    missing_prompt = 1 if prompt_rate is None else 0
    score = -1.0 if prompt_rate is None else float(prompt_rate)
    return (missing_prompt, -score, loss_tok, loss_total)


def _build_markdown(summary: dict) -> str:
    args = summary["args"]
    lines: list[str] = []
    lines.append("# OneFlow Text-Only Checkpoint Sweep Summary")
    lines.append("")
    lines.append(f"- ckpt_root: `{args['ckpt_root']}`")
    lines.append(f"- dataset_dir: `{args['dataset_dir']}`")
    lines.append(f"- prompts_file: `{args['prompts_file']}`")
    lines.append(f"- ranking_mode: `{args['ranking_mode']}`")
    lines.append(f"- checkpoints_evaluated: {summary['num_checkpoints']}")
    best = summary.get("best_checkpoint")
    if best:
        lines.append(f"- best_checkpoint: `{best['name']}`")
        lines.append(f"- best_checkpoint_path: `{best['path']}`")
    lines.append("")
    lines.append("| rank | checkpoint | step | prompt_pass_rate | loss_tok_mean | loss_total_mean |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rec in summary["results"]:
        p = rec["prompt"]["expected_pass_rate"]
        p_str = "N/A" if p is None else f"{100.0 * float(p):.1f}%"
        lines.append(
            f"| {rec['rank']} | {rec['name']} | {rec['step'] if rec['step'] is not None else 'final'} | "
            f"{p_str} | {rec['loss']['trained']['aggregate']['loss_tok']['mean']:.4f} | "
            f"{rec['loss']['trained']['aggregate']['loss_total']['mean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = transformers.HfArgumentParser((Args,))
    (args,) = parser.parse_args_into_dataclasses()

    if not args.ckpt_root:
        raise ValueError("--ckpt_root is required")
    if not args.dataset_dir:
        raise ValueError("--dataset_dir is required")
    if not os.path.exists(LOSS_EVAL_SCRIPT):
        raise FileNotFoundError(f"Missing script: {LOSS_EVAL_SCRIPT}")
    if bool(args.run_prompt_eval) and not os.path.exists(PROMPT_EVAL_SCRIPT):
        raise FileNotFoundError(f"Missing script: {PROMPT_EVAL_SCRIPT}")

    ckpt_root = _abs_path(args.ckpt_root)
    dataset_dir = _abs_path(args.dataset_dir)
    prompts_file = _abs_path(args.prompts_file)
    output_dir = _abs_path(args.output_dir) if args.output_dir else os.path.join(ckpt_root, "stable_eval")
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = _discover_checkpoints(args)
    if not checkpoints:
        raise ValueError(
            f"No checkpoints found under {ckpt_root} "
            "(try --include_final True or relax --min_step/--max_step)."
        )

    prompt_device = str(args.prompt_device or args.device)
    records: list[dict] = []
    for row in checkpoints:
        ckpt_name = row["name"]
        ckpt_path = row["path"]
        ckpt_out = os.path.join(output_dir, ckpt_name)
        os.makedirs(ckpt_out, exist_ok=True)

        loss_json = os.path.join(ckpt_out, "loss_report.json")
        _run_cmd(
            [
                sys.executable,
                "-u",
                LOSS_EVAL_SCRIPT,
                "--model_dir",
                ckpt_path,
                "--dataset_dir",
                dataset_dir,
                "--device",
                str(args.device),
                "--batch_size",
                str(args.loss_batch_size),
                "--num_batches",
                str(args.loss_num_batches),
                "--seed",
                str(args.loss_seed),
                "--seeds",
                str(args.loss_seeds),
                "--compare_random",
                "True" if bool(args.loss_compare_random) else "False",
                "--output_json",
                loss_json,
            ]
        )
        with open(loss_json, "r", encoding="utf-8") as f:
            loss_report = json.load(f)

        prompt_stats = {
            "total_rows": 0,
            "expected_rows": 0,
            "pass_rows": 0,
            "expected_pass_rate": None,
            "output_dir": None,
        }
        if bool(args.run_prompt_eval):
            prompt_out = os.path.join(ckpt_out, "prompt_eval")
            cmd = [
                sys.executable,
                "-u",
                PROMPT_EVAL_SCRIPT,
                "--model_dir",
                ckpt_path,
                "--prompts_file",
                prompts_file,
                "--output_dir",
                prompt_out,
                "--device",
                prompt_device,
                "--seed",
                str(args.prompt_seed),
                "--visualize",
                "True" if bool(args.prompt_visualize) else "False",
                "--visualize_terminal",
                "True" if bool(args.prompt_visualize_terminal) else "False",
                "--visualize_save_html",
                "True" if bool(args.prompt_visualize_save_html) else "False",
                "--visualize_save_json",
                "True" if bool(args.prompt_visualize_save_json) else "False",
                "--visualize_color_map",
                str(args.prompt_visualize_color_map),
            ]
            if int(args.prompt_max_prompts) > 0:
                cmd.extend(["--max_prompts", str(int(args.prompt_max_prompts))])
            _run_cmd(cmd)
            prompt_jsonl = os.path.join(prompt_out, "report.jsonl")
            if not os.path.exists(prompt_jsonl):
                raise FileNotFoundError(f"Missing prompt report: {prompt_jsonl}")
            prompt_stats = _load_prompt_pass_stats(prompt_jsonl)
            prompt_stats["output_dir"] = prompt_out

        records.append(
            {
                "name": ckpt_name,
                "path": ckpt_path,
                "step": row["step"],
                "is_final": bool(row["is_final"]),
                "loss": loss_report,
                "prompt": prompt_stats,
            }
        )

    ranked = sorted(records, key=lambda r: _record_sort_key(r, args.ranking_mode))
    for i, rec in enumerate(ranked, start=1):
        rec["rank"] = int(i)

    summary = {
        "args": {
            "ckpt_root": ckpt_root,
            "dataset_dir": dataset_dir,
            "prompts_file": prompts_file,
            "output_dir": output_dir,
            "device": str(args.device),
            "loss_batch_size": int(args.loss_batch_size),
            "loss_num_batches": int(args.loss_num_batches),
            "loss_seed": int(args.loss_seed),
            "loss_seeds": str(args.loss_seeds),
            "loss_compare_random": bool(args.loss_compare_random),
            "run_prompt_eval": bool(args.run_prompt_eval),
            "prompt_device": prompt_device,
            "prompt_seed": int(args.prompt_seed),
            "prompt_max_prompts": int(args.prompt_max_prompts),
            "prompt_visualize": bool(args.prompt_visualize),
            "prompt_visualize_terminal": bool(args.prompt_visualize_terminal),
            "prompt_visualize_save_html": bool(args.prompt_visualize_save_html),
            "prompt_visualize_save_json": bool(args.prompt_visualize_save_json),
            "prompt_visualize_color_map": str(args.prompt_visualize_color_map),
            "include_final": bool(args.include_final),
            "min_step": int(args.min_step),
            "max_step": int(args.max_step),
            "ranking_mode": str(args.ranking_mode),
        },
        "num_checkpoints": int(len(ranked)),
        "best_checkpoint": ranked[0] if ranked else None,
        "results": ranked,
    }

    out_json = os.path.join(output_dir, "summary.json")
    out_md = os.path.join(output_dir, "summary.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(summary))

    print(json.dumps({"summary_json": out_json, "summary_md": out_md}, indent=2))
    if ranked:
        best = ranked[0]
        print(
            f"[BEST] {best['name']} "
            f"step={best['step']} "
            f"prompt_pass_rate={best['prompt']['expected_pass_rate']} "
            f"loss_tok_mean={best['loss']['trained']['aggregate']['loss_tok']['mean']:.4f} "
            f"loss_total_mean={best['loss']['trained']['aggregate']['loss_total']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()

