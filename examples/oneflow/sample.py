"""
Example:
python -u examples/oneflow/sample.py --model_dir "models/oneflow/checkpoint-final" --prompt "Once upon a time"
"""

from dataclasses import dataclass

import torch
import transformers

import dllm
from dllm.pipelines.oneflow.models import OneFlowModel
from dllm.pipelines.oneflow.sampler import OneFlowSampler, OneFlowSamplerConfig, OneFlowSamplerOutput


@dataclass
class ScriptArguments:
    model_dir: str = "models/oneflow/checkpoint-final"
    seed: int = 42
    prompt: str = ""


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sampler = OneFlowSampler(model=model, tokenizer=tokenizer)

    prompt_ids = tokenizer.encode(script_args.prompt, add_special_tokens=False)
    out = sampler.sample([prompt_ids], sampler_args, return_dict=True)

    assert isinstance(out, OneFlowSamplerOutput)
    text = tokenizer.decode(out.sequences[0].tolist(), skip_special_tokens=False)
    print("\n=== Text ===")
    print(text)

    if out.images:
        print("\n=== Images (latents) ===")
        for i, (lat, ti) in enumerate(zip(out.images, out.image_times or [])):
            print(f"[image {i}] latent shape={tuple(lat.shape)} t={ti:.3f}")


if __name__ == "__main__":
    main()


