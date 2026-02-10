"""
Example:
python -u examples/oneflow/sample.py --model_dir "models/oneflow/checkpoint-final" --prompt "Once upon a time"
"""

from dataclasses import dataclass

import torch
import transformers

import dllm
from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.runtime_config import (
    DEFAULT_TEXT_SAMPLER_RUNTIME,
    load_runtime_config,
    resolve_section_settings,
)
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ScriptArguments:
    model_dir: str = "models/oneflow/checkpoint-final"
    seed: int = 42
    prompt: str = ""
    device: str = "auto"  # auto|cpu|cuda|npu
    skip_special_tokens: bool = True


@dataclass
class SamplerArgs(OneFlowSamplerConfig):
    dt: float | None = None
    max_steps: int | None = None
    image_num_tokens: int = 64
    temperature: float | None = None
    use_pi_gate: bool | None = None
    max_w: float | None = None
    condition_text_on_time: bool | None = None
    kappa_scheduler_cls: str | None = None
    return_dict: bool = True


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerArgs))
    script_args, sampler_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_dir)
    model = OneFlowModel.from_pretrained(script_args.model_dir, map_location="cpu").eval()

    def _npu_available() -> bool:
        return bool(
            hasattr(torch, "npu")
            and hasattr(torch.npu, "is_available")
            and torch.npu.is_available()
        )

    dev = str(getattr(script_args, "device", "auto") or "auto").lower()
    if dev == "auto":
        if _npu_available():
            device = torch.device("npu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif dev in ("npu", "cuda", "cpu"):
        device = torch.device(dev)
    else:
        raise ValueError(f"Unknown --device: {script_args.device} (expected auto|cpu|cuda|npu)")

    model = model.to(device)

    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)
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
        logger.warning(
            "No oneflow_runtime_config.json found in model_dir; using built-in sampler defaults."
        )
    if applied_ckpt_keys:
        logger.info(f"Using checkpoint runtime config for sampler keys: {sorted(applied_ckpt_keys)}")
    if override_keys:
        logger.warning(
            f"CLI overrides checkpoint runtime config for sampler keys: {sorted(override_keys)}"
        )

    sampler_args.kappa_scheduler_cls = str(resolved_sampling["scheduler_cls"])
    sampler_args.dt = float(resolved_sampling["dt"])
    sampler_args.max_steps = int(resolved_sampling["max_steps"])
    sampler_args.temperature = float(resolved_sampling["temperature"])
    sampler_args.use_pi_gate = bool(resolved_sampling["use_pi_gate"])
    sampler_args.append_only = bool(resolved_sampling["append_only"])
    sampler_args.edit_prompt = bool(resolved_sampling["edit_prompt"])
    sampler_args.condition_text_on_time = bool(resolved_sampling["condition_text_on_time"])
    sampler_args.max_w = (
        None if resolved_sampling["max_w"] is None else float(resolved_sampling["max_w"])
    )

    prompt_ids = tokenizer.encode(script_args.prompt, add_special_tokens=False)
    out = sampler.sample([prompt_ids], sampler_args, return_dict=True)

    assert isinstance(out, OneFlowSamplerOutput)
    text = tokenizer.decode(
        out.sequences[0].tolist(),
        skip_special_tokens=bool(getattr(script_args, "skip_special_tokens", True)),
    )
    print("\n=== Text ===")
    print(text)

    if out.images:
        print("\n=== Images (latents) ===")
        for i, (lat, ti) in enumerate(zip(out.images, out.image_times or [])):
            print(f"[image {i}] latent shape={tuple(lat.shape)} t={ti:.3f}")


if __name__ == "__main__":
    main()


