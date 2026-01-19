"""
Interactive sampling for OneFlow.

Example:
python -u examples/oneflow/chat.py --model_dir "models/oneflow/checkpoint-final"
"""

import sys
from dataclasses import dataclass

import torch
import transformers

from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput


@dataclass
class ScriptArguments:
    model_dir: str = "models/oneflow/checkpoint-final"
    seed: int = 42
    device: str = "auto"  # auto|cpu|cuda|npu
    skip_special_tokens: bool = True


@dataclass
class SamplerArgs(OneFlowSamplerConfig):
    dt: float = 0.05
    max_steps: int = 256
    image_num_tokens: int = 64
    temperature: float = 0.0
    use_pi_gate: bool = True
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

    print("OneFlow chat. Type a prompt and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("User> ").strip("\n")
        except EOFError:
            break
        if prompt is None:
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        out = sampler.sample([prompt_ids], sampler_args, return_dict=True)
        assert isinstance(out, OneFlowSamplerOutput)

        text = tokenizer.decode(
            out.sequences[0].tolist(),
            skip_special_tokens=bool(getattr(script_args, "skip_special_tokens", True)),
        )
        print("\nAssistant>\n" + text + "\n")

        if out.images:
            print(f"[debug] sampled {len(out.images)} image latents")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)


