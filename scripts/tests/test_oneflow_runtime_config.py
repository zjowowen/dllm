from __future__ import annotations

import tempfile
from types import SimpleNamespace

from dllm.core.schedulers import LinearKappaScheduler
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    build_runtime_config,
    load_runtime_config,
    resolve_section_settings,
    save_runtime_config,
)


def test_build_runtime_config_tracks_training_knobs():
    args = SimpleNamespace(
        scheduler_cls="CosineKappaScheduler",
        tau_text_max=1.7,
        condition_text_on_time=True,
        text_loss_type="paper",
    )
    sched = LinearKappaScheduler()
    runtime_cfg = build_runtime_config(training_args=args, scheduler=sched)

    assert runtime_cfg["training"]["scheduler_cls"] == "LinearKappaScheduler"
    assert abs(float(runtime_cfg["training"]["tau_text_max"]) - 1.7) < 1e-8
    assert runtime_cfg["training"]["condition_text_on_time"] is True
    assert runtime_cfg["sampling"]["scheduler_cls"] == "LinearKappaScheduler"
    assert runtime_cfg["sampling"]["condition_text_on_time"] is True
    assert float(runtime_cfg["sampling"]["max_w"]) == 20.0


def test_resolve_section_settings_precedence():
    runtime_cfg = {
        "sampling": {
            "scheduler_cls": "CosineKappaScheduler",
            "dt": 0.1,
            "max_steps": 128,
            "temperature": 0.3,
            "use_pi_gate": False,
            "append_only": True,
            "edit_prompt": False,
            "condition_text_on_time": True,
            "max_w": 15.0,
        }
    }
    resolved, override_keys, applied_ckpt_keys = resolve_section_settings(
        runtime_config=runtime_cfg,
        section_name="sampling",
        cli_overrides={
            "scheduler_cls": None,
            "dt": 0.05,  # explicit CLI override
            "max_steps": None,
            "temperature": None,
            "use_pi_gate": None,
            "append_only": None,
            "edit_prompt": None,
            "condition_text_on_time": None,
            "max_w": None,
        },
        defaults=DEFAULT_TEXT_SAMPLER_RUNTIME,
    )

    assert abs(float(resolved["dt"]) - 0.05) < 1e-8
    assert int(resolved["max_steps"]) == 128
    assert bool(resolved["use_pi_gate"]) is False
    assert bool(resolved["append_only"]) is True
    assert "dt" in override_keys
    assert "max_steps" in applied_ckpt_keys


def test_load_runtime_config_fallback_to_sibling_checkpoint_final():
    with tempfile.TemporaryDirectory() as td:
        ckpt_final = f"{td}/checkpoint-final"
        ckpt_step = f"{td}/checkpoint-100"
        save_runtime_config(
            ckpt_final,
            {
                "version": 1,
                "training": {"scheduler_cls": "LinearKappaScheduler"},
                "sampling": {"scheduler_cls": "LinearKappaScheduler", "dt": 0.05},
            },
        )
        # No runtime config under checkpoint-100; loader should fallback to sibling checkpoint-final.
        loaded = load_runtime_config(ckpt_step)
        assert loaded is not None
        assert loaded["sampling"]["scheduler_cls"] == "LinearKappaScheduler"
