from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Mapping

from dllm.core.schedulers import BaseKappaScheduler

RUNTIME_CONFIG_FILENAME = "oneflow_runtime_config.json"
RUNTIME_CONFIG_VERSION = 1

# Keep these defaults aligned with OneFlow sampler/trainer defaults.
DEFAULT_TRAINING_RUNTIME: dict[str, Any] = {
    "scheduler_cls": "LinearKappaScheduler",
    "tau_text_max": 2.0,
    "condition_text_on_time": False,
    "text_loss_type": "paper",
}

DEFAULT_TEXT_SAMPLER_RUNTIME: dict[str, Any] = {
    "scheduler_cls": "LinearKappaScheduler",
    "dt": 0.05,
    "max_steps": 512,
    "temperature": 0.0,
    "use_pi_gate": True,
    "append_only": False,
    "edit_prompt": False,
    "condition_text_on_time": False,
    "max_w": 20.0,
}

DEFAULT_TEXT_EVAL_RUNTIME: dict[str, Any] = {
    "scheduler_cls": "LinearKappaScheduler",
    "tau_text_max": 2.0,
    "condition_text_on_time": False,
}


def scheduler_class_name(
    scheduler: BaseKappaScheduler | None,
    *,
    fallback: str | None = None,
) -> str | None:
    if scheduler is None:
        return fallback
    return type(scheduler).__name__


def _scheduler_init_kwargs(scheduler: BaseKappaScheduler | None) -> dict[str, Any]:
    if scheduler is None:
        return {}
    if dataclasses.is_dataclass(scheduler):
        return dataclasses.asdict(scheduler)
    return {}


def build_runtime_config(
    *,
    training_args: Any,
    scheduler: BaseKappaScheduler | None,
    sampler_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scheduler_cls = scheduler_class_name(
        scheduler,
        fallback=str(
            getattr(training_args, "scheduler_cls", DEFAULT_TRAINING_RUNTIME["scheduler_cls"])
            or DEFAULT_TRAINING_RUNTIME["scheduler_cls"]
        ),
    )
    tau_text_max = float(
        getattr(training_args, "tau_text_max", DEFAULT_TRAINING_RUNTIME["tau_text_max"])
        or DEFAULT_TRAINING_RUNTIME["tau_text_max"]
    )
    condition_text_on_time = bool(
        getattr(
            training_args,
            "condition_text_on_time",
            DEFAULT_TRAINING_RUNTIME["condition_text_on_time"],
        )
    )
    text_loss_type = str(
        getattr(training_args, "text_loss_type", DEFAULT_TRAINING_RUNTIME["text_loss_type"])
        or DEFAULT_TRAINING_RUNTIME["text_loss_type"]
    )

    sampling_cfg = dict(DEFAULT_TEXT_SAMPLER_RUNTIME)
    if sampler_defaults:
        sampling_cfg.update(dict(sampler_defaults))
    # The most important consistency knobs should follow training values by default.
    sampling_cfg["scheduler_cls"] = scheduler_cls
    sampling_cfg["condition_text_on_time"] = condition_text_on_time

    return {
        "version": int(RUNTIME_CONFIG_VERSION),
        "training": {
            "scheduler_cls": scheduler_cls,
            "scheduler_init_kwargs": _scheduler_init_kwargs(scheduler),
            "tau_text_max": tau_text_max,
            "condition_text_on_time": condition_text_on_time,
            "text_loss_type": text_loss_type,
        },
        "sampling": sampling_cfg,
    }


def save_runtime_config(output_dir: str, runtime_config: Mapping[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, RUNTIME_CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(runtime_config), f, ensure_ascii=False, indent=2)
    return path


def resolve_runtime_config_path(model_dir: str) -> str:
    model_dir = os.path.abspath(str(model_dir))
    direct = os.path.join(model_dir, RUNTIME_CONFIG_FILENAME)
    if os.path.exists(direct):
        return direct

    parent = os.path.dirname(model_dir)
    sibling_final = os.path.join(parent, "checkpoint-final", RUNTIME_CONFIG_FILENAME)
    if os.path.exists(sibling_final):
        return sibling_final

    parent_cfg = os.path.join(parent, RUNTIME_CONFIG_FILENAME)
    if os.path.exists(parent_cfg):
        return parent_cfg

    return direct


def load_runtime_config(model_dir: str) -> dict[str, Any] | None:
    path = resolve_runtime_config_path(model_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        out = json.load(f)
    if not isinstance(out, dict):
        raise ValueError(f"Invalid runtime config format at {path}: expected a JSON object.")
    return out


def resolve_section_settings(
    *,
    runtime_config: Mapping[str, Any] | None,
    section_name: str,
    cli_overrides: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Merge settings with precedence:
      CLI explicit override > checkpoint runtime config > hard default.

    Returns:
      (resolved, cli_override_keys, checkpoint_applied_keys)
    """
    section: Mapping[str, Any] = {}
    if runtime_config is not None:
        raw = runtime_config.get(section_name, {})
        if isinstance(raw, Mapping):
            section = raw

    resolved: dict[str, Any] = {}
    cli_override_keys: list[str] = []
    checkpoint_applied_keys: list[str] = []

    for key, default_val in defaults.items():
        cli_val = cli_overrides.get(key, None)
        has_cli = cli_val is not None
        has_ckpt = key in section and section.get(key, None) is not None
        ckpt_val = section.get(key, None) if has_ckpt else None

        if has_cli:
            resolved[key] = cli_val
            if has_ckpt and cli_val != ckpt_val:
                cli_override_keys.append(str(key))
            continue

        if has_ckpt:
            resolved[key] = ckpt_val
            checkpoint_applied_keys.append(str(key))
            continue

        resolved[key] = default_val

    return resolved, cli_override_keys, checkpoint_applied_keys
